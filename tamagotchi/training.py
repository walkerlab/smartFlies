import time
import numpy as np
import itertools
import torch
import pandas as pd
import os
from collections import deque
from a2c_ppo_acktr.storage import RolloutStorage
# from tamagotchi.eval import eval_lite
from a2c_ppo_acktr import utils

def update_by_schedule(envs, schedule_dict, curr_step):
    # TODO: implement a density function over the probs of selecting a new TC value
    # updating the env_collection will change the currently running envs :o
    
    updated = False
    for k, _schedule_dict in schedule_dict.items():
        if curr_step in _schedule_dict: # see if the current step should be updated 
            updated = k
            # different update functions since birthx is managed by each process, while wind_cond is managed by my custom SubprocVecEnv
            if k == 'birthx': # update the birthx value in the envs. Sparsity is decided in each envs.reset() at each trial
                envs.env_method_apply_to_all("update_env_param", {k: _schedule_dict[curr_step]})
                print(f"update_env_param {k}: {_schedule_dict[curr_step]} at {curr_step}")
            elif k == 'wind_cond': 
                envs.update_wind_direction(_schedule_dict[curr_step])
                print(f"update_env_param {k}: {_schedule_dict[curr_step]} at {curr_step}")
            else:
                raise NotImplementedError
    return updated # return the course that is updated, if any


def build_tc_schedule_dict(total_number_periods, interleave=True, **kwargs):
    """Builds a training curriculum schedule dictionary. 
    Args:
        total_number_periods: number of updates 
        **kwargs: A dictionary containing the schedule information. 
            Each key is an env variable and each value is a dict that specifies how the env variable should be scheduled.
            num_classes: number of steps to take to reahc max difficulty.
            difficulty range: a list of two values, the first is the starting value, the second is the max value.
            dtype: the type of the env variable.
            step_type: the type of step to take to reach the max value. Either 'log' or 'linear'.
    
    Returns:
        schedule_dict: A dictionary of dicts. Each key is an env variable to be updated according to its schedule information.
            The schedule information is stored a dict with key: at which "episode" to update, and value: update the value to what.
            
            
    """
    # initialize the default course directory 
    course_dirctory = {'birthx': {'num_classes': 6, 'difficulty_range': [0.7,0.001], 'dtype': 'float', 'step_type': 'log'}, 
                   'wind_cond': {'num_classes': 2, 'difficulty_range': [1, 3], 'dtype': 'int', 'step_type': 'linear'}} 
    if not interleave:
        # if not interleave, do the two lessons consecutively, with the wind first 
        course_dirctory = {'wind_cond': {'num_classes': 2, 'difficulty_range': [1, 3], 'dtype': 'int', 'step_type': 'linear'},
                           'birthx': {'num_classes': 6, 'difficulty_range': [0.7,0.001], 'dtype': 'float', 'step_type': 'log'}} 
    
        
    # update the default course directory with the input kwargs
    for course in kwargs:
        now_kwargs = kwargs[course]
        for k in now_kwargs:
            if course_dirctory[course][k] != now_kwargs[k]:
                course_dirctory[course][k] = now_kwargs[k]
                print(f"Updated {course} {k} = {course_dirctory[course][k]} to {now_kwargs[k]}", flush=True)

    # calculate the scheduled value for each class, given the number of updates and the difficulty range
    tmp_schedule = []
    for course in course_dirctory:
        now_course = course_dirctory[course]
        # set the scheduled diffuclty value for each class
        assert len(now_course['difficulty_range']) == 2
        if now_course['step_type'] == 'log':
            scheduled_value = np.logspace(np.log10(now_course['difficulty_range'][0]),
                                        np.log10(now_course['difficulty_range'][1]),
                                        now_course['num_classes'] +1 , endpoint=True) 
        elif now_course['step_type'] == 'linear':
            scheduled_value = np.linspace(now_course['difficulty_range'][0],
                                    now_course['difficulty_range'][1],
                                    now_course['num_classes'] +1 , endpoint=True) 
        else: 
            raise ValueError("step_type must be 'log' or 'linear'")
        now_course['scheduled_value'] = scheduled_value 
        tmp_schedule.append(zip(itertools.cycle([course]), scheduled_value[1:])) # the lower bound is the first value -> set at the beginning
        
    
    if interleave: # interleave the scheduled values for each class
        course_schedule = itertools.chain.from_iterable(itertools.zip_longest(*tmp_schedule))
        course_schedule = [c for c in course_schedule if c]
    else: # do the two lessons consecutively
        course_schedule = list(itertools.chain(*tmp_schedule))

    # build the schedule dictionary - when/how to update the env variable according to the course schedule
    schedule_dict = {}
    for k in course_dirctory.keys():
        schedule_dict[k] = {}
    total_num_classes = len(course_schedule) + 1 # +1 since the first step is at 0, which is not an actual step... 
    when_2_update = np.linspace(0, total_number_periods, total_num_classes, endpoint=False, dtype=int)
    when_2_update = when_2_update[1:] # take out the first step which is at 0, not an actual step
    
    for course in course_dirctory.keys():
        schedule_dict[course][0] = course_dirctory[course]['scheduled_value'][0]
    for i, course in enumerate(course_schedule):
        course_name, scheduled_value = course
        schedule_dict[course_name][when_2_update[i]] = scheduled_value

    print("[DEBUG] schedule_dict:", schedule_dict)
    return schedule_dict

def log_episode(training_log, j, total_num_steps, start, episode_rewards, episode_puffs, episode_plume_densities, episode_wind_directions, num_updates):
    # update the training log with the current episode's statistics
    end = time.time()
    print(
        "Update {}/{}, T {}, FPS {}, {}-training-episode: mean/median {:.1f}/{:.1f}, \
            min/max {:.1f}/{:.1f}, std {:.2f}, num_puffs mean/std {:.2f}/{:.2f}, \
            plume_densities {:.2f}/{:.2f}, wind_dir mean/std {:.2f}/{:.2f} "
        .format(j, num_updates, 
                total_num_steps,
                int(total_num_steps / (end - start)),
                len(episode_rewards), np.mean(episode_rewards),
                np.median(episode_rewards), 
                np.min(episode_rewards),
                np.max(episode_rewards),
                np.std(episode_rewards),                    
                np.mean(episode_puffs),
                np.std(episode_puffs),
                np.mean(episode_plume_densities),
                np.std(episode_plume_densities),
                np.mean(episode_wind_directions),
                np.std(episode_wind_directions),)) 

    training_log.append({
            'update': j,
            'total_updates': num_updates,
            'T': total_num_steps,
            'FPS': int(total_num_steps / (end - start)),
            'window': len(episode_rewards), 
            'mean': np.mean(episode_rewards),
            'median': np.median(episode_rewards), 
            'min': np.min(episode_rewards),
            'max': np.max(episode_rewards),
            'std': np.std(episode_rewards),
            'num_puffs_mean': np.mean(episode_puffs),
            'num_puffs_std': np.std(episode_puffs),
            'plume_density_mean': np.mean(episode_plume_densities),
            'plume_density_std': np.std(episode_plume_densities),
            'wind_direction_mean': np.mean(episode_wind_directions),
            'wind_direction_std': np.std(episode_wind_directions),
        })
    return training_log

def training_loop(agent, envs, args, device, actor_critic, 
    training_log=None, eval_log=None, eval_env=None, rollouts=None):
    ##############################################################################################################
    # setting up
    ##############################################################################################################
    if not rollouts: 
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs[0].observation_space.shape, 
                        envs[0].action_space,
                        actor_critic.recurrent_hidden_state_size)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes # args.num_env_steps 20M for all # args.num_steps=2048 (found in logs) # args.num_processes=4=mini_batch (found in logs)
    
    episode_rewards = deque(maxlen=50) 
    episode_plume_densities = deque(maxlen=50)
    episode_puffs = deque(maxlen=50)
    episode_wind_directions = deque(maxlen=50)
    
    best_mean = 0.0

    training_log = training_log if training_log is not None else []
    eval_log = eval_log if eval_log is not None else []
    
    # initialize the curriculum schedule
    if args.birthx_linear_tc_steps:
        schedule = build_tc_schedule_dict(num_updates, birthx={'num_classes': args.birthx_linear_tc_steps, 
                                                                'difficulty_range': [0.7, args.birthx], 
                                                                'dtype': 'float', 'step_type': 'linear'}, 
                                          wind_cond={'num_classes': 2, 'difficulty_range': [1, 3], 
                                                     'dtype': 'int', 'step_type': 'linear'}) # wind_cond: 1 is constant, 2 is switch, 3 is noisy
        update_by_schedule(envs, schedule, 0) # update the initialized envs according to the curriculum schedule. The default init values are incorrect, hence this update s.t. reset() returns correctly.
    
    obs = envs.reset()
    rollouts.obs[0].copy_(obs) # https://discuss.pytorch.org/t/which-copy-is-better/56393
    rollouts.to(device)
    start = time.time()
    
    # at each bout of update
    for j in range(num_updates):
        print(f"On update {j} of {num_updates}")
        start1 = time.time()

        # decrease learning rate linearly
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)
        
        # update envs according to the curriculum schedule
        if args.birthx_linear_tc_steps:
            updated = update_by_schedule(envs, schedule, j)
            if updated: # save model if an update to env occurred during this trial
                lesson_fpath = args.model_fpath.replace(".pt", f'_before_{updated}{schedule[updated][j]}_update{j}.pt')
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None),
                    agent.optimizer.state_dict(),
                ], lesson_fpath)
                print('Saved', lesson_fpath)

        
        ##############################################################################################################
        # at each step of training 
        ##############################################################################################################
        for step in range(args.num_steps):
            if step < 3:
                start2 = time.time()
            
            # Sample actions ~0.008 seconds
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, activities = actor_critic.act(
                    rollouts.obs[step], 
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            obs, reward, done, infos = envs.step(action)
            
            if step < 3:
                end2 = time.time()
                # print(f"Step {step} took {end2-start2} seconds")
            
            # save and log
            for i, d in enumerate(done): # if done, log the episode info. Care about what kind of env is encountered
                if d:
                    try:
                        # Note: only ouput these to infos when done
                        episode_rewards.append(infos[i]['episode']['r'])
                        episode_plume_densities.append(infos[i]['plume_density']) # plume_density and num_puffs are expected to be similar across different agents. Tracking to confirm. 
                        episode_puffs.append(infos[i]['num_puffs'])
                        episode_wind_directions.append(envs.ds2wind(infos[i]['dataset'])) # density and dataset are logged to see eps. statistics implemented by the curriculum
                    except KeyError:
                        raise KeyError("Logging info not found... check why it's not here when done")

            # If done then clean the history of observations in the recurrent_hidden_states. Done in the MLPBase forward method
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            # TODO unsure what this is [may be about if done and self.env._max_episode_steps == self.env._elapsed_steps:]
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            if step < 3:
                start3 = time.time()
                
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks) # ~0.0006s
            if step < 3:
                end3 = time.time()
                # print(f"Step {step} took {end3-start3} seconds")
        ##############################################################################################################
        # UPDATE AGENT ~ 0.48s
        ##############################################################################################################
        start4 = time.time()
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        end4 = time.time()
        # print(f"Update {j} took {end4-start4} seconds")
        ##############################################################################################################
        # save for every interval-th episode or for the last epoch
        ##############################################################################################################
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None),
                agent.optimizer.state_dict(),
            ], args.model_fpath)
            print('Saved', args.model_fpath)

            current_mean = np.median(episode_rewards)
            if current_mean >= best_mean:
                best_mean = current_mean
                fname = f'{args.model_fpath}.best'
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], fname)
                print('Saved', fname)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            training_log = log_episode(training_log, j, total_num_steps, start, episode_rewards, episode_puffs, episode_plume_densities, episode_wind_directions, num_updates)
            # Save training curve
            pd.DataFrame(training_log).to_csv(args.training_log)


        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            if eval_env is not None:
                eval_record = eval_lite(agent, eval_env, args, device, actor_critic, )
                eval_record['T'] = total_num_steps
                eval_log.append(eval_record)
                print("eval_lite:", eval_record)

                save_path = args.save_dir
                os.makedirs(save_path, exist_ok=True)
                fname = os.path.join(save_path, f'{args.env_name}_{args.outsuffix}_eval.csv')
                pd.DataFrame(eval_log).to_csv(fname)
        end1 = time.time()
        # print(f"Update {j} took {end1-start1} seconds")
                
    return training_log, eval_log
