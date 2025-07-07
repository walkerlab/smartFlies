import torch
import numpy as np
import pickle
import gc
import argparse
import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152 # this line is from #https://stackoverflow.com/questions/37893755/tensorflow-set-cuda-visible-devices-within-jupyter
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import eval.agent_analysis as agent_analysis
import eval.log_analysis as log_analysis


def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    
def log_perturb_analysis(log_dict, **kwargs):
    for key, value in kwargs.items():
        if key == 'ls_activity' or key == 'ls_activity_perturbed':
            log_dict[key].append({
                    'rnn_hxs': value['rnn_hxs'].cpu().numpy().squeeze(),
                    'hx1_actor': value['hx1_actor'].cpu().numpy().squeeze(),
                    'hx1_critic': value['hx1_critic'].cpu().numpy().squeeze(), 
                    'hidden_actor': value['hidden_actor'].cpu().numpy().squeeze(), 
                    'hidden_critic': value['hidden_critic'].cpu().numpy().squeeze(), 
                    'value': value['value'].cpu().numpy().squeeze()} )
        else:
            log_dict[key].append(value)
            

def save_log_to_pkl(episode_logs, out_dir, f_prefix):
    # save the episode_logs to a pkl file
    with open(f"{out_dir}/{f_prefix}_perturb_analysis.pkl", 'wb') as f:
        pickle.dump({episode_logs}, f)


def open_perturb_loop(traj_df_stacked, stacked_neural_activity, actor_critic, orthogonal_basis, sigma_noise, args):
    traj_df_stacked.reset_index(drop=True, inplace=True)
    episode_logs = [] # list of dfs
    all_eps = traj_df_stacked['ep_idx'].unique()
    all_eps.sort()
    subset_eps = all_eps[args.from_eps:args.to_eps+1] # get from_eps to to_eps
    for eps in subset_eps:
        if args.verbose: fprint(f"[LOG] starting episode {eps} at {datetime.datetime.now()}")

        # for each episode - get the trajectory and neural activity data 
        idx = traj_df_stacked[traj_df_stacked['ep_idx'] == eps].index
        # get the trajectory at these idx
        now_traj = traj_df_stacked.loc[idx]
        now_traj.reset_index(drop=True, inplace=True)
        # get the neural activity at these idx
        now_activity = stacked_neural_activity[idx]
        masks = torch.zeros(1, 1, device=args.device) # set to 0 at t = -1, or at terminal state
        masks.fill_(1.0) # always set to 1 since the first documented traj row is at t = 0 and we are comparing up to Terminal-1
        tidx_range = now_traj.tidx.max()
        
        with torch.no_grad():
            for rep in range(args.perturb_rep):
                episode_log = {'ls_tidx': [],
                    'ls_dist': [],
                    'ls_dist_perturbed': [],
                    'ls_KL_divergence': [],
                    'ls_h_perturbed': [],
                    'ls_perturb_by': [],
                    'ls_activity': [],
                    'ls_activity_perturbed': []}
                if args.verbose>1: print(f"perturb rep {rep}")
                args.seed += 1
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                for timestep in range(tidx_range): # compare up to t-1
                    if args.verbose>1: print(f"timestep {timestep}/{tidx_range}")
                    # load eps data at timestep
                    now_traj_row = now_traj.iloc[timestep] # get the df row at this timestep
                    now_activity_row = now_activity[timestep]
                    obs = get_obs_from_traj_row(now_traj_row, device=args.device)
                    recurrent_hidden_states = get_activity_from_activity_row(now_activity_row, device=args.device)
                    
                    # get the unperturbed action at t
                    value, action, _, recurrent_hidden_states_next, activity = actor_critic.act(
                        obs,
                        recurrent_hidden_states, 
                        masks, 
                        deterministic=True) # un-perturbed action
                    
                    # perturb the agent & get the perturbed action at t 
                    recurrent_hidden_states_perturbed, perturb_by = agent_analysis.perturb_rnn_activity(recurrent_hidden_states, 
                                                                                                        orthogonal_basis, sigma_noise, 
                                                                                                        args.perturb_RNN_by, 
                                                                                                        sample_noise_by=args.sample_noise_by, 
                                                                                                        return_perturb_by=True)
                    value_perturbed, action_perturbed, _, recurrent_hidden_states_perturbed_next, activity_perturbed = actor_critic.act(
                        obs,
                        recurrent_hidden_states_perturbed, 
                        masks, 
                        deterministic=True)
                    
                    # get a divergence measure
                    dist = actor_critic.dist(activity['hidden_actor'])
                    dist_perturbed = actor_critic.dist(activity_perturbed['hidden_actor'])
                    KL_divergence = torch.distributions.kl.kl_divergence(torch.distributions.Normal(dist.mean, dist.stddev),
                                                                        torch.distributions.Normal(dist_perturbed.mean, dist_perturbed.stddev))
                    # NOTE: dist.mean==dist.mode, dist.mode -> action -> squash action i.e. (np.tanh(action) + 1)/2; see env.py
                    log_perturb_analysis(episode_log, 
                                        ls_tidx=timestep, 
                                        ls_dist=[dist.mean.cpu().numpy().squeeze(), dist.stddev.cpu().numpy().squeeze()], # mean.shape = [1,2]; first is speed & second is angular velocity
                                        ls_dist_perturbed=[dist_perturbed.mean.cpu().numpy().squeeze(), dist_perturbed.stddev.cpu().numpy().squeeze()],
                                        ls_KL_divergence=KL_divergence.cpu().numpy().squeeze(),
                                        ls_h_perturbed=recurrent_hidden_states_perturbed.cpu().numpy().squeeze(),
                                        ls_perturb_by=perturb_by.cpu().numpy().squeeze(),
                                        ls_activity=activity,
                                        ls_activity_perturbed=activity_perturbed)
                    
                    # every 20 timesteps, do a test to make sure OL step result still agreed with closed loop record
                    if timestep % 20 == 0:
                        recurrent_hidden_states_next_record = get_activity_from_activity_row(now_activity[timestep + 1], device=args.device)
                        action_next = get_action_from_traj_row(now_traj.iloc[timestep + 1], device=args.device)
                        if not np.allclose(recurrent_hidden_states_next.cpu().detach().numpy(), recurrent_hidden_states_next_record.cpu().detach().numpy(), atol=1e-4):
                            fprint(f"[BEEP] eps {eps} rep {rep}: recurrent_hidden_states not equal at timestep {timestep}, and the biggest difference is {np.max(np.abs(recurrent_hidden_states_next.cpu().detach().numpy() - recurrent_hidden_states_next_record.cpu().detach().numpy()))}")
                            fprint(f"[BEEP] eps {eps} rep {rep}: obs {obs}, masks {masks}")
                    if args.verbose>1: print(f"timestep end {timestep}/{tidx_range}")
                if args.verbose>1: print(f"all ts done for perturb rep {rep}")
                episode_logs.append({'ep_idx' : eps, 
                                    'perturb_by': args.perturb_RNN_by,
                                    'rep': rep,
                                    'seed': args.seed,
                                    'episode_log':episode_log})
            gc.collect()
    return episode_logs

def get_obs_from_traj_row(now_row, device='cpu'):
    obs = tuple([now_row['wind_x_obs'], now_row['wind_y_obs'], 
       now_row['odor_raw'], # odor obs as is (not rectified, etc)
       now_row['agent_angle_x'], now_row['agent_angle_y'], 
       now_row['ego_course_direction_x'], now_row['ego_course_direction_y']])
    obs = [obs]
    obs = torch.tensor(obs).float().to(device)
    return obs

def get_activity_from_activity_row(now_row, device='cpu'):
    activity = torch.tensor([now_row]).float().to(device)
    return activity

def get_action_from_traj_row(now_row, device='cpu'):
    action = [now_row['step'], now_row['turn']]
    action = torch.tensor(action).float().to(device)
    return action

def save_log_to_pkl(episode_logs, out_dir, f_prefix):
    # save the episode_logs to a pkl file
    with open(f"{out_dir}/{f_prefix}.pkl", 'wb') as f:
        pickle.dump(episode_logs, f)

class Args(argparse.Namespace):
    seed = 137
    device = 'cpu'
    # device = 'cuda:0'
    flip_ventral_optic_flow = False
    perturb_RNN_by_ortho_set = False
    perturb_RNN_by = False
    no_vec_norm_stats = True
    model_fname = None  # Assuming no default is given
    out_dir = None
    ###
    from_eps = 20
    to_eps = 39


if __name__ == '__main__':
    # exactly the same as evalCli.py argument loading, with additions from traj_analysis.py
    args=Args()
    args.apparent_wind = True
    args.visual_feedback = True
    np.random.seed(args.seed)
  
    args.model_fname = '/src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_ALL_noisy_wind_0.001/weights/plume_951_23354e57874d619687478a539a360146.pt'
    args.out_dir = 'open_loop_perturbation'
    args.f_prefix = os.path.basename(args.model_fname).replace(".pt", "") # eg: plume_seed_hash
    args.f_dir = os.path.dirname(args.model_fname) # f_dir should follow {/path/to/experiment}/weights
    exp_dir = os.path.dirname(args.f_dir) # {/path/to/experiment}
    args.abs_out_dir = '/'.join([exp_dir, args.out_dir, args.f_prefix]) # {/path/to/experiment}/{args.out_dir=eval}/plume_seed_hash/

    args.model_fname = '/src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_ALL_noisy_wind_0.001/weights/plume_951_23354e57874d619687478a539a360146.pt'
    args.eval_folder = 'eval'
    args.dataset = 'noisy3x5b5'
    args.eval_folder = args.model_fname.replace('weights', args.eval_folder).replace('.pt', '/')
    args.verbose = 0
    fprint(f"Trajectory directory: {args.eval_folder}")
    fprint(f"Output directory: {args.abs_out_dir}")
    
    args.from_eps = int(sys.argv[1])
    args.to_eps = int(sys.argv[2])
    if len(sys.argv) > 3:
        args.sample_noise_by = sys.argv[3]
        args.out_dir = f"{args.out_dir}_{args.sample_noise_by}"
        print(f"Sample noise by {args.sample_noise_by}, and output to {args.out_dir}")
    fprint(f"Starting from {args.from_eps} to {args.to_eps}")
    # make sure the directory exists
    os.makedirs('/'.join([exp_dir, args.out_dir]), exist_ok=True)
    os.makedirs(args.abs_out_dir, exist_ok=True)

    try:
        actor_critic, obs_rms, optimizer_state_dict = torch.load(args.model_fname, map_location=torch.device(args.device))
    except ValueError:
        actor_critic, obs_rms = torch.load(args.model_fname, map_location=torch.device(args.device))
        
    
    selected_df = log_analysis.get_selected_df(args.eval_folder, [args.dataset],
                                            n_episodes_home=240,
                                            n_episodes_other=240,  
                                            balanced=False,
                                            oob_only=False,
                                            verbose=True)

    traj_df_stacked, stacked_neural_activity = log_analysis.get_traj_and_activity_and_stack_them(selected_df, 
                                                                                                obtain_neural_activity = True, 
                                                                                                obtain_traj_df = True, 
                                                                                                get_traj_tmp = False) # get_traj_tmp False to get normalized obersevations - feed to agent for action
    
    # check if all counts are 1
    assert (traj_df_stacked.groupby(['ep_idx','tidx']).value_counts() == 1).all()
    # check if all lines up
    assert (len(stacked_neural_activity) == len(traj_df_stacked))
    # TODO important arg
    args.perturb_rep = 100
    args.perturb_RNN_by = 'subspace_WN_in_nullspace'
    args.perturb_RNN_by_ortho_set = '/src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_ALL_noisy_wind_0.001/eval/plume_951_23354e57874d619687478a539a360146/ranked_orthogonal_basis_and_var_with_wind_encoding_subspace_951.pkl'
    if args.perturb_RNN_by == 'subspace' or args.perturb_RNN_by == 'nullspace' or args.perturb_RNN_by == 'subspace_WN_in_nullspace':
        file_content = agent_analysis.import_orthogonal_basis(args.perturb_RNN_by_ortho_set) # 64x64, where the first row is the wind encoding subspace
        if len(file_content) == 1:
            orthogonal_basis = file_content[0]
            sigma_noise = 0.01 # variance in the wind encoding direction 
        elif len(file_content) == 2:
            orthogonal_basis = file_content[0]
            sigma_noise = file_content[1] # variance in each orthogonal basis direction - shape of (64,)
        fprint(f"Loading orthogonal basis from {args.perturb_RNN_by_ortho_set}, and perturbing hidden states with noise {sigma_noise}")
    elif args.perturb_RNN_by == 'all':
        orthogonal_basis='dummy'
    episode_logs = open_perturb_loop(traj_df_stacked, stacked_neural_activity, actor_critic, orthogonal_basis, sigma_noise, args)
    f_prefix = f"OL_perturb_{args.perturb_RNN_by}_{args.from_eps}_{args.to_eps}_{args.perturb_rep}rep"
    fprint(f"Saving perturbation analysis to {args.abs_out_dir}/{f_prefix}.pkl")
    save_log_to_pkl(episode_logs, args.abs_out_dir, f_prefix)