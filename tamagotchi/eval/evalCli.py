from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import traceback

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152 # this line is from #https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
import sys

import torch
import argparse
import numpy as np
import torch
import pandas as pd
import pickle

import matplotlib 
matplotlib.use("Agg")

import numpy as np
from env import make_vec_envs
import agent_analysis 
import os
# import log_analysis # for viz hidden trajectories

### UTILITY ###
def evaluate_agent(actor_critic, env, args):
    num_tests = 0
    reward_total_sum = 0
    episode_logs = []
    episode_summaries = []
    venv = env.unwrapped.envs[0].venv

    # Fixed evaluation assay
    # Fixed eval meshgrid:
    # 2 timestamps X 3 x-locations X 5 y-locations X 8 angles = 240 runs       
    if args.fixed_eval:
        venv.loc_algo = 'fixed'
        venv.angle_algo = 'fixed'
        venv.time_algo = 'fixed' 
        
        grids = []
        # Initializing the agent 
        # x coordinate is fixed at venv.fixed_x
        for venv.fixed_x in [4.0, 6.0, 8.0]:
          for venv.fixed_time_offset in [0.0, 1.0]: # time_offset
            env.reset()

            # Figure out extent of plume in y-direction
            Z = venv.get_abunchofpuffs()
            X_mean, X_var = venv.fixed_x, 0.5 # fixed_x +/- band
            Z = pd.DataFrame(Z)
            Yqs = Z.query("(x >= (@X_mean - @X_var)) and (x <= (@X_mean + @X_var))")['y'].quantile([0.0,1.0]).to_numpy()
            y_min, y_max = Yqs[0], Yqs[1]

            y_stray = 0.5
            # y coordinate depends on the dataset
            if ('switch' in args.dataset) or ('noisy' in args.dataset):
                y = np.linspace(y_min, y_max, 5) # Don't start outside plume
            else: 
                y = np.linspace(y_min, y_max, 3) # Can start off plume 
                y = np.concatenate([ [y_min - y_stray], y, [y_max + y_stray] ]) # 5 now
            # face direction of the agent 
            a = np.linspace(0, 2, 9)[:8]*np.pi # angle

            # print(f"At {venv.t_val}s: y_min {y_min}, y_max {y_max}; Grid: {y}")
            grid = np.meshgrid(y, a)
            grid = np.array(grid).reshape(2, -1).T # [loc_y, angle]

            # Add loc_x & time 
            grid = pd.DataFrame(grid)
            grid.columns = ['loc_y', 'angle']
            grid['loc_x'] = venv.fixed_x
            grid['time'] = venv.fixed_time_offset

            grids.append(grid)
        # make a DF of starting locations for the agent
        grid = pd.concat(grids).to_numpy() # Stack
        args.test_episodes = grid.shape[0]  # TODO HACK test_episodes never used. Take this out or make functional.
        print(f"Using fixed evaluation sequence [time, angle, loc_y]... ({args.test_episodes} episodes) ")


    for i_episode in range(args.test_episodes):

        if args.fixed_eval:
            venv.fixed_y = grid[i_episode, 0] # meters
            venv.fixed_angle = grid[i_episode, 1] # radians
            venv.fixed_x = grid[i_episode, 2] # meters
            venv.fixed_time_offset = grid[i_episode, 3] # seconds

        # recurrent_hidden_states = torch.zeros(1, 
                    # actor_critic.recurrent_hidden_state_size, device='cpu')
        recurrent_hidden_states = torch.zeros(1, 
                    actor_critic.recurrent_hidden_state_size, device=args.device)
        # masks = torch.zeros(1, 1, device='cpu')
        masks = torch.zeros(1, 1, device=args.device)
        obs = env.reset()

        reward_sum = 0    
        ep_step = 0
        trajectory = []
        observations = []
        actions = []
        rewards = []
        infos = []
        activities = []


        while True:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states, activity = actor_critic.act(
                    obs, 
                    recurrent_hidden_states, 
                    masks, 
                    deterministic=True)

            obs, reward, done, info = env.step(action)
            masks.fill_(0.0 if done else 1.0)

            if args.device != 'cpu':
                _obs = obs.to("cpu").numpy()
                _reward = reward.to("cpu").numpy().squeeze()
                _action = action.to("cpu").numpy().squeeze()
            else:
                _obs = obs.detach().numpy()
                _reward = reward.detach().numpy().squeeze()
                _action = action.detach().numpy().squeeze()
            _info = info[0]
            _done = done

            if info[0]['done'] != 'HOME':
                if _reward>9:
                    print(f"Reward: {_reward}, info: {_info}; WRONG REWARD! Should NOT have added 100.")
            elif info[0]['done'] == 'HOME':
                if _reward<=9:
                    print(f"Reward: {_reward}, info: {_info}; WRONG REWARD! Should HAVE added 100.")

            _reward = (_reward + 100) if _reward > 9 else _reward # HACK! Unsure/Debug!
            
            reward_sum += _reward
            # NOTE: Squash action. Whetehr post squash'd actions are documented... This mattered for viz. 04/08/24.
                # vaguely remember changing post-squash actions to be documented. Check git blame to make sure... 
            # print(f"[DEBUG] Documented Action pre-squash: {_action}", flush=True, file=sys.stderr)
            if args.squash_action:
                action = (np.tanh(action.to("cpu")) + 1)/2
            # print(f"[DEBUG] Squashed Action: {action}", flush=True, file=sys.stderr)
            trajectory.append( _info['location'] )
            observations.append( _obs )
            # print(f"[DEBUG] Documented Action: {_action}", flush=True, file=sys.stderr)
            actions.append( _action )
            rewards.append( _reward )
            infos.append( [_info] )
            if args.device != 'cpu':
                activities.append( {
                    'rnn_hxs': activity['rnn_hxs'].to("cpu").numpy().squeeze(),
                    'hx1_actor': activity['hx1_actor'].to("cpu").numpy().squeeze(),
                    # 'hx1_critic': activity['hx1_critic'].detach().numpy().squeeze(), # don't care
                    # 'hidden_actor': activity['hidden_actor'].detach().numpy().squeeze(), # don't care
                    # 'hidden_critic': activity['hidden_critic'].detach().numpy().squeeze(), # don't care
                    'value': activity['value'].to("cpu").numpy().squeeze(),
                } )
            else:
                activities.append( {
                    'rnn_hxs': activity['rnn_hxs'].detach().numpy().squeeze(),
                    'hx1_actor': activity['hx1_actor'].detach().numpy().squeeze(),
                    # 'hx1_critic': activity['hx1_critic'].detach().numpy().squeeze(), # don't care
                    # 'hidden_actor': activity['hidden_actor'].detach().numpy().squeeze(), # don't care
                    # 'hidden_critic': activity['hidden_critic'].detach().numpy().squeeze(), # don't care
                    'value': activity['value'].detach().numpy().squeeze(),
                } )

            ep_step += 1

            if _done:
                num_tests += 1
                reward_total_sum += reward_sum
                reward_mean = reward_total_sum / num_tests
                break

        n_steps = len(trajectory)
        print(f"Episode: {i_episode}, reward_sum: {reward_sum}, Steps: {n_steps}, Reason: {_info['done']}")


        episode_log = {
            'trajectory': trajectory,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'infos': infos,
            'activity': activities,
        }
        episode_logs.append(episode_log)

        episode_summary = {
            'idx': i_episode,
            'reward_sum': reward_sum.item(),
            'n_steps': n_steps,
            'reason': _info['done'],
        }
        episode_summaries.append(episode_summary)


    return episode_logs, episode_summaries


### BACK TO MAIN ###
def eval_loop(args, actor_critic, test_sparsity=True):
    try:
        # Common output directory
        OUTPREFIX = args.model_fname.replace(".pt", "/")
        os.makedirs(OUTPREFIX, exist_ok=True)
        kwargs = {}
        if args.apparent_wind: # don't pass in if don't care
            kwargs = {'apparent_wind': args.apparent_wind}
        #### ------- Nonsparse ------- #### 
        env = make_vec_envs(
            args.env_name,
            args.seed + 1000,
            num_processes=1,
            gamma=0.99, # redundant
            log_dir=None, # redundant
            # device='cpu',
            device=args.device,
            args=args,
            allow_early_resets=False,
            **kwargs)

        if 'switch' in args.dataset: 
            venv = env.unwrapped.envs[0].venv
            venv.qvar = 0.0
            venv.t_val_min = 58.0
            venv.reset_offset_tmax = 3.0
            venv.diff_max = 0.9
            venv.reload_dataset()

        episode_logs, episode_summaries = evaluate_agent(actor_critic, env, args)

        fname3 = f"{OUTPREFIX}/{args.dataset}.pkl"
        with open(fname3, 'wb') as f_handle:
            pickle.dump(episode_logs, f_handle)
            print("Saving", fname3)

        fname3 = f"{OUTPREFIX}/{args.dataset}_summary.csv"
        pd.DataFrame(episode_summaries).to_csv(fname3)
        print("Saving", fname3)

        # graph_OUTPREFIX = f"{OUTPREFIX}/eg_trajectory/"
        zoom = 1 if 'constant' in args.dataset else 2    
        zoom = 3 if args.walking else zoom
        if not args.no_viz:
            agent_analysis.visualize_episodes(episode_logs[:args.viz_episodes], 
                                            zoom=zoom, 
                                            dataset=args.dataset,
                                            animate=False, # Quick plot
                                            fprefix=args.dataset,
                                            diffusionx=args.diffusionx,
                                            outprefix=OUTPREFIX
                                            )

        # for episode_idx in range(len(episode_logs[:args.viz_episodes])):
        #     log = episode_logs[episode_idx]
        #     if actor_critic.is_recurrent:
        #         ep_activity = pd.DataFrame(log['activity'])['rnn_hxs'].to_list()
        #     else:
        #         ep_activity = pd.DataFrame(log['activity'])['hx1_actor'].to_list()

        #     traj_df = pd.DataFrame( log['trajectory'] )
        #     traj_df['t_val'] = [record[0]['t_val'] for record in log['infos']]
        #     log_analysis.animate_activity_1episode(ep_activity, 
        #             traj_df, 
        #             episode_idx, 
        #             fprefix=args.dataset,
        #             outprefix=OUTPREFIX,
        #             pca_dims=3)

    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()


    #### ------- Sparse ------- #### #### ------- Sparse ------- #### #### ------- Sparse ------- #### #### ------- Sparse ------- #### 
    if test_sparsity:
        x = np.linspace(0.9, 0.1, 5)
        y = np.array([0.05, 0.01, 0.005, 0.001, 0.0005])
        z = np.concatenate((x,y))
        # for birthx in np.arange(0.9, 0.1, -0.05):
        for birthx in z:
            birthx = round(birthx, 4)
            print("Sparse/birthx:", birthx)
            try:
                args.birthx_max = birthx # load time birthx: subsample the plume data at the time of loading 
                args.birthx = 1.0 # dynamic birthx: subsample by rand.unif.[birthx, 1] at the time of reset at each epoch, on top of the loaded birthx
                args.loc_algo = 'quantile'
                args.diff_max = 0.8
                args.movex = 1

                env = make_vec_envs(
                    args.env_name,
                    args.seed + 1000,
                    num_processes=1,   
                    gamma=0.99, # redundant
                    log_dir=None, # redundant
                    # device='cpu',
                    device=args.device,
                    args=args,
                    allow_early_resets=False,
                    **kwargs)

                episode_logs, episode_summaries = evaluate_agent(actor_critic, env, args)

                fname3 = f"{OUTPREFIX}/{args.dataset}_{birthx}.pkl"
                with open(fname3, 'wb') as f_handle:
                    pickle.dump(episode_logs, f_handle)
                    print("Saving", fname3)

                fname3 = f"{OUTPREFIX}/{args.dataset}_{birthx}_summary.csv"
                pd.DataFrame(episode_summaries).to_csv(fname3)
                print("Saving", fname3)


                zoom = 1 if 'constant' in args.dataset else 2    
                zoom = 3 if args.walking else zoom
                if not args.no_viz:
                    if birthx in [0.1, 0.05, 0.001]:
                        agent_analysis.visualize_episodes(episode_logs[:args.viz_episodes], 
                            zoom=zoom, 
                            dataset=args.dataset,
                            animate=True,
                            fprefix=f'sparse_{args.dataset}_{birthx}', 
                            outprefix=OUTPREFIX,
                            diffusionx=args.diffusionx,
                            birthx=birthx,
                            )

                # for episode_idx in range(len(episode_logs[:args.viz_episodes])):
                #     log = episode_logs[episode_idx]
                #     if actor_critic.is_recurrent:
                #         ep_activity = pd.DataFrame(log['activity'])['rnn_hxs'].to_list()
                #     else:
                #         ep_activity = pd.DataFrame(log['activity'])['hx1_actor'].to_list()

                #     traj_df = pd.DataFrame( log['trajectory'] )
                #     traj_df['t_val'] = [record[0]['t_val'] for record in log['infos']]

                #     log_analysis.animate_activity_1episode(ep_activity, 
                #             traj_df, 
                #             episode_idx, 
                #             fprefix=f'sparse_{args.dataset}_{birthx}',
                #             outprefix=OUTPREFIX,
                #             pca_dims=3)

            except Exception as e:
                print(f"Exception: {e}")
                traceback.print_exc()





### MAIN ###
if __name__ == "__main__":
    # TODO: Needs updating
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--algo', default='ppo')
    parser.add_argument('--dataset', default='constantx5b5')
    parser.add_argument('--model_fname')
    parser.add_argument('--test_episodes', type=int, default=100)
    parser.add_argument('--viz_episodes', type=int, default=10)
    parser.add_argument('--no_viz', type=bool, default=True)
    # parser.add_argument('--viz_episodes', type=int, default=10)

    parser.add_argument('--fixed_eval', action='store_true', default=False)
    parser.add_argument('--test_sparsity', action='store_true', default=False)
    parser.add_argument('--device', default="cuda:0")
    
    # env related
    parser.add_argument('--diffusionx',  type=float, default=1.0)
    parser.add_argument('--apparent_wind',  type=bool, default=False)


    args = parser.parse_args()
    print(args)
    args.det = True # override

    np.random.seed(args.seed)
    args.env_name = 'plume'
    args.env_dt = 0.04
    args.turnx = 1.0
    args.movex = 1.0
    args.birthx = 1.0
    args.loc_algo = 'quantile'
    args.time_algo = 'uniform'
    args.diff_max = 0.8
    args.diff_min = 0.8
    args.auto_movex = False
    args.auto_reward = False
    args.wind_rel = True
    args.action_feedback = False
    # args.action_feedback = True
    args.walking = False
    args.radiusx = 1.0
    args.r_shaping = ['step'] # redundant
    args.rewardx = 1.0
    args.squash_action = True

    args.diffusion_min = args.diffusionx
    args.diffusion_max = args.diffusionx

    args.flipping = False
    args.odor_scaling = False
    args.qvar = 0.0
    args.stray_max = 2.0
    args.birthx_max = 1.0
    args.masking = None
    args.stride = 1
    args.obs_noise = 0.0
    args.act_noise = 0.0

    args.dynamic = False

    args.recurrent_policy = True if ('GRU' in args.model_fname) or ('RNN' in args.model_fname) else False
    args.rnn_type = 'VRNN' if 'RNN' in args.model_fname else 'GRU'

    args.stacking = 0
    if 'MLP' in args.model_fname:
        args.stacking = int( args.model_fname.split('MLP_s')[-1].split('_')[0] )
    
    # actor_critic, ob_rms, optimizer_state_dict = torch.load(args.model_fname, map_location=torch.device('cpu'))
    try:
        actor_critic, ob_rms, optimizer_state_dict = torch.load(args.model_fname, map_location=torch.device(args.device))
    except ValueError:
        actor_critic, ob_rms = torch.load(args.model_fname, map_location=torch.device(args.device))
        
    eval_loop(args, actor_critic, test_sparsity=args.test_sparsity)