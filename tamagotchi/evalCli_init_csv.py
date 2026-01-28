# option to read in a csv file with the initial conditions for the agent
# for testing true wind agents on Toha's dataset
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import traceback

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152 # this line is from #https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
import inspect
import torch
import argparse
import numpy as np
import torch
import pandas as pd
import pickle

import matplotlib 
matplotlib.use("Agg")

import numpy as np
from env import make_vec_envs, PlumeEnvironment_v2, PlumeEnvironment_v3
import eval.agent_analysis as agent_analysis
import os
# import log_analysis # for viz hidden trajectories


def load_init_conditions(csv_file):
    # Read in the csv file
    init_conditions = pd.read_csv(csv_file)
    # Convert to dictionary
    if init_conditions.shape[1] == 7:
        # Check number of columns
        init_conditions.columns = ['file', 'loc_x', 'loc_y', 'time', 't_move', 'plume_growth_rate', 'plume_density']
    elif init_conditions.shape[1] == 9:
        init_conditions.columns = ['file', 'loc_x', 'loc_y', 'time', 't_move', 'plume_growth_rate', 'plume_density', 'initial_distance', 'plume_shape']
    else:
        init_conditions.columns = ['loc_x', 'loc_y', 'time', 'file']
    random_HD = False
    if random_HD:
        # initialize head direction in radians
        init_head_directions = np.linspace(0, 2, 9)[:8]*np.pi # angle - taken from eval_Cli.py
        # build a new column with the distinct values
        # Repeat the DataFrame for each head direction
        repeated_df = pd.concat([init_conditions] * len(init_head_directions), ignore_index=True)
        # Assign the distinct values to the new column
        repeated_df['angle'] = np.repeat(init_head_directions, len(init_conditions))
        print(f"[DEBUG] PEv3: repeated_df angles {repeated_df['file'].values}", flush=True)
    else: # fixed HD towards the origin
        init_head_directions = np.arctan2(-init_conditions['loc_y'], -init_conditions['loc_x']) # Initial heading towards the origin (0,0)
        repeated_df = init_conditions.copy()
        repeated_df['angle'] = init_head_directions.values
        print(f"[DEBUG] PEv3: fixed_df angles {repeated_df['file'].values}", flush=True)
    # Convert to numpy array
    if repeated_df.shape[1] != 5:
        out_ndarry = repeated_df[['loc_y', 'angle', 'loc_x', 'time', 'file']].to_numpy() # column order follows the prev. convention
    else:
        out_ndarry = repeated_df[['loc_y', 'angle', 'loc_x', 'time', 'file']].to_numpy() # column order follows the prev. convention

    return out_ndarry


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
        if args.init_conditions:
            grid = load_init_conditions(args.init_conditions)
        else: 
            raise ValueError("This script is explicitly for testing initial conditions in a csv file.")
        # make a DF of starting locations for the agent
        args.test_episodes = grid.shape[0]  # TODO HACK test_episodes never used. Take this out or make functional.
        print(f"Using fixed evaluation sequence [time, angle, loc_y]... ({args.test_episodes} episodes) ")


    for i_episode in range(args.test_episodes):
        if args.fixed_eval:
            venv.fixed_y = grid[i_episode, 0] # meters
            venv.fixed_angle = grid[i_episode, 1] # radians
            venv.fixed_x = grid[i_episode, 2] # meters
            venv.t_val_min = grid[i_episode, 3] # seconds
            venv.t_val_max = venv.t_val_min + venv.reset_offset_tmax + 1.0*venv.episode_steps_max/venv.fps + 1.00
            venv.set_dataset(grid[i_episode, 4]) # read in the dataset file


        recurrent_hidden_states = torch.zeros(1, 
                    actor_critic.recurrent_hidden_state_size, device=args.device)
        masks = torch.zeros(1, 1, device=args.device)
        if args.perturb_along_subspace:
            sigma_noise = 0.01
            if os.path.isfile(args.perturb_along_subspace):
                print(f"Loading orthogonal basis from {args.perturb_along_subspace}, and perturbing hidden states with noise {sigma_noise}")
                orthogonal_basis = agent_analysis.import_orthogonal_basis(args.perturb_along_subspace) # 64x64, where the first row is the wind encoding subspace
                recurrent_hidden_states = agent_analysis.perturb_rnn_activity(recurrent_hidden_states, orthogonal_basis, sigma_noise, 'subspace')
            elif args.perturb_along_subspace == 'all':
                print(f"Perturbing all hidden states with noise {sigma_noise}")
                orthogonal_basis='dummy'
                recurrent_hidden_states = agent_analysis.perturb_rnn_activity(recurrent_hidden_states, orthogonal_basis, sigma_noise, 'all')
            elif args.perturb_along_subspace == 'subspace':
                raise ValueError("Need to provide a file that stores the orthogonal basis for the wind encoding subspace.")
            else:
                raise ValueError(f"Unsupported value for args.perturb_along_subspace {args.perturb_along_subspace}.")
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
                if args.perturb_along_subspace:
                    if os.path.isfile(args.perturb_along_subspace):
                        recurrent_hidden_states = agent_analysis.perturb_rnn_activity(recurrent_hidden_states, orthogonal_basis, sigma_noise, 'subspace')
                    elif args.perturb_along_subspace == 'all':
                        recurrent_hidden_states = agent_analysis.perturb_rnn_activity(recurrent_hidden_states, orthogonal_basis, sigma_noise, 'all')
                    else:
                        raise ValueError(f"Unsupported value for args.perturb_along_subspace {args.perturb_along_subspace}.")
                    
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
            if args.squash_action:
                action = (np.tanh(action.to("cpu")) + 1)/2
            trajectory.append( _info['location'] )
            observations.append( _obs )
            actions.append( _action )
            rewards.append( _reward )
            infos.append( [_info] )
            if args.device != 'cpu':
                if args.perturb_along_subspace:
                    activities.append( {
                        'rnn_hxs': activity['rnn_hxs'].to("cpu").numpy().squeeze(), # returned from a2c_ppo_acktr.model.MLPBase.forward; a copy of the hidden states
                        'rnn_hxs_perturbed': recurrent_hidden_states.to("cpu").numpy().squeeze(), # recurrent_hidden_states object has been perturbed 
                        'hx1_actor': activity['hx1_actor'].to("cpu").numpy().squeeze(),
                        'value': activity['value'].to("cpu").numpy().squeeze(),
                    } )
                else:
                    activities.append( {
                        'rnn_hxs': activity['rnn_hxs'].to("cpu").numpy().squeeze(), # returned from a2c_ppo_acktr.model.MLPBase.forward; a copy of the hidden states
                        'hx1_actor': activity['hx1_actor'].to("cpu").numpy().squeeze(),
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
        # Extract default values from __init__ method
        init_signature_vis_fb_params = inspect.signature(PlumeEnvironment_v3.__init__)
        init_signature = inspect.signature(PlumeEnvironment_v2.__init__)
        defaults = {param.name: param.default for param in init_signature.parameters.values() if param.default != inspect.Parameter.empty}
        defaults.update({param.name: param.default for param in init_signature_vis_fb_params.parameters.values() if param.default != inspect.Parameter.empty})
        # Write into args if not already present
        for key, value in defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)
        if not args.no_vec_norm_stats:
            vecNormalize_pkl_file = args.model_fname.replace('.pt', '_vecNormalize.pkl')
            kwargs = {'vecNormalize_pkl_file': vecNormalize_pkl_file, 'eval': True} # init settings file path and mode eval to be true 
        else:
            kwargs = {}
        
        
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
            **kwargs
            )
        venv = env.unwrapped.envs[0].venv
        venv.episode_steps_max = args.sim_steps_max # not currently in make_vec_envs kwargs so set explicitly, fine for now 110625. 
        
        if 'switch' in args.dataset: 
            venv.qvar = 0.0
            venv.t_val_min = 58.0
            venv.reset_offset_tmax = 3.0
            venv.diff_max = 0.9
            venv.reload_dataset()

        episode_logs, episode_summaries = evaluate_agent(actor_critic, env, args)

        fname3 = f"{args.abs_out_dir}/{args.dataset}.pkl"
        with open(fname3, 'wb') as f_handle:
            pickle.dump(episode_logs, f_handle)
            print("Saving", fname3)

        fname3 = f"{args.abs_out_dir}/{args.dataset}_summary.csv"
        pd.DataFrame(episode_summaries).to_csv(fname3)
        print("Saving", fname3)

        # graph_abs_out_dir = f"{args.abs_out_dir}/eg_trajectory/"
        if not args.no_viz:
            zoom = 1 if 'constant' in args.dataset else 2    
            zoom = 3 if args.walking else zoom
            agent_analysis.visualize_episodes(episode_logs[:args.viz_episodes], 
                                            zoom=zoom, 
                                            dataset=args.dataset,
                                            animate=False, # Quick plot
                                            fprefix=args.dataset,
                                            diffusionx=args.diffusionx,
                                            abs_out_dir=args.abs_out_dir
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
        #             abs_out_dir=abs_out_dir,
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
            # print("Sparse/birthx:", birthx)
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
                    allow_early_resets=False)

                episode_logs, episode_summaries = evaluate_agent(actor_critic, env, args)

                fname3 = f"{args.abs_out_dir}/{args.dataset}_{birthx}.pkl"
                with open(fname3, 'wb') as f_handle:
                    pickle.dump(episode_logs, f_handle)
                    print("Saving", fname3)

                fname3 = f"{args.abs_out_dir}/{args.dataset}_{birthx}_summary.csv"
                pd.DataFrame(episode_summaries).to_csv(fname3)
                print("Saving", fname3)

                if not args.no_viz:
                    zoom = 1 if 'constant' in args.dataset else 2    
                    zoom = 3 if args.walking else zoom
                    if birthx in [0.1, 0.05, 0.001]:
                        agent_analysis.visualize_episodes(episode_logs[:args.viz_episodes], 
                            zoom=zoom, 
                            dataset=args.dataset,
                            animate=True,
                            fprefix=f'sparse_{args.dataset}_{birthx}', 
                            abs_out_dir=args.abs_out_dir,
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
                #             abs_out_dir=args.abs_out_dir,
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
    parser.add_argument('--model_fname', help='absolute path to model')
    parser.add_argument('--test_episodes', type=int, default=100)
    parser.add_argument('--viz_episodes', type=int, default=10)
    parser.add_argument('--no_viz', type=bool, default=True)
    # parser.add_argument('--viz_episodes', type=int, default=10)

    parser.add_argument('--fixed_eval', action='store_true', default=False)
    parser.add_argument('--test_sparsity', action='store_true', default=False)
    parser.add_argument('--device', default="cuda:0")
    
    # env related
    parser.add_argument('--diffusionx',  type=float, default=1.0)
    parser.add_argument('--apparent_wind', type=bool, default=False)
    parser.add_argument('--visual_feedback', type=bool, default=False)
    parser.add_argument('--flip_ventral_optic_flow', type=bool, default=False) # for eval to see the behavioral impact of flipping course direction perception.
    parser.add_argument('--perturb_along_subspace', type=str, default=False, help='a file that stores a orthogonal basis, where the first vector is the wind encoding subspace')
    parser.add_argument('--no_vec_norm_stats', action='store_true', default=False, help='an agent that is trained without storing vecNormalize stats, or did not use it during training')
    parser.add_argument('--out_dir', type=str, default='eval')
    parser.add_argument('--init_conditions', type=str, default=False, help='a csv file that stores the initial conditions for the agent. for testing toha')
    parser.add_argument('--verbose', type=int, default=0, help='verbose for the env obj')
    parser.add_argument('--t_val_min', type=int, default=60, help='where to start loading the plume data from in seconds')
    parser.add_argument('--sim_steps_max', type=int, default=299, help='maximum number of simulation steps')

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
    args.walking = False
    args.radiusx = 1.0
    args.r_shaping = ['step'] 
    args.rewardx = 1.0
    args.squash_action = True

    args.diffusion_min = args.diffusionx
    args.diffusion_max = args.diffusionx

    args.flipping = False
    args.odor_scaling = False
    args.qvar = 0.0
    args.stray_max = 50.0 # 121525 - enlarge for toha's large arena - inits up to 27m from nearest - just gonna turn off oob for now (turn on by adding oob to r_shaping)
    args.birthx_max = 1.0
    args.masking = None
    args.stride = 1
    args.obs_noise = 0.0
    args.act_noise = 0.0

    args.dynamic = False

    args.stacking = 0
    args.recurrent_policy = True # used in setting up network in main. Outside of that, used once in make_env (always true in my case)
    
    args.f_prefix = os.path.basename(args.model_fname).replace(".pt", "") # eg: plume_seed_hash
    args.f_dir = os.path.dirname(args.model_fname) # f_dir should follow {/path/to/experiment}/weights
    exp_dir = os.path.dirname(args.f_dir) # {/path/to/experiment}

    args.abs_out_dir = '/'.join([exp_dir, args.out_dir, args.f_prefix]) # {/path/to/experiment}/{args.out_dir=eval}/plume_seed_hash/
    
    print(f"Output directory: {args.abs_out_dir}")
    # make sure the directory exists
    os.makedirs('/'.join([exp_dir, args.out_dir]), exist_ok=True)
    os.makedirs(args.abs_out_dir, exist_ok=True)
    
    # actor_critic, ob_rms, optimizer_state_dict = torch.load(args.model_fname, map_location=torch.device('cpu'))
    try:
        actor_critic, obs_rms, optimizer_state_dict = torch.load(args.model_fname, map_location=torch.device(args.device), weights_only=False)
    except ValueError:
        actor_critic, obs_rms = torch.load(args.model_fname, map_location=torch.device(args.device), weights_only=False)
        
    eval_loop(args, actor_critic, test_sparsity=args.test_sparsity)