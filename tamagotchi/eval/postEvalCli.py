from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""

import torch
import argparse
import os
import sys
import numpy as np
import torch
from natsort import natsorted

import traceback

import matplotlib 
matplotlib.use("Agg")


import numpy as np
# import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import agent_analysis
import os
import log_analysis
import arch_utils as archu
import sklearn.decomposition as skld


def smart_outprefix(model_dir, dataset, out_reldir):
    # makes a directory for videos, create dir if not present
    outprefix = '/'.join([model_dir, out_reldir, dataset])
    os.makedirs(outprefix, exist_ok=True)
    return outprefix


def post_eval(args):
    is_recurrent = True
    selected_df = log_analysis.get_selected_df(args.model_dir, 
                                  args.use_datasets, 
                                  n_episodes_home=60, 
                                  n_episodes_other=60,
                                  min_ep_steps=0)
    # Generate common PCA
    h_episodes = []
    traj_dfs = []
    squash_action = True

    for episode_log in selected_df['log']:
        ep_activity = log_analysis.get_activity(episode_log, 
            is_recurrent, do_plot=False)
        h_episodes.append(ep_activity)
                
    h_episodes_stacked = np.vstack(h_episodes)

    pca_common = skld.PCA(3, whiten=False)
    pca_common.fit(h_episodes_stacked)

    # Get neural net

    model_fname = args.model_dir.rstrip("/").replace("eval", "weights") + ".pt" # plume_951_23354e57874d619687478a539a360146.pt should be
    # check if file exists
    if not os.path.exists(model_fname):
        print(f"Model file {model_fname} not found. Skipping...")
    else:
        print(f"Loading model from {model_fname}")
        try:
            actor_critic, ob_rms, optimizer_state_dict= \
                torch.load(model_fname, map_location=torch.device('cpu'), weights_only=False)
        except ValueError:
            actor_critic, ob_rms= \
                torch.load(model_fname, map_location=torch.device('cpu'),  weights_only=False)
        net = actor_critic.base.rnn #.weight_hh_l0.detach().numpy()
        J0 = net.weight_hh_l0.detach().numpy()

    if args.viz_wind_reg:
        # fit a linear regression that predicts wind angle from neural activity
        reg = agent_analysis.fit_regression_from_neural_activity_to_latent(selected_df, \
            'wind_angle_ground_theta', stacked_neural_activity = h_episodes_stacked) 

    if args.viz_only_these_episodes:
        print(f"Visualizing episodes {args.viz_only_these_episodes}...")
        subset_df = selected_df[selected_df['idx'].isin(args.viz_only_these_episodes)]
        print(f"Found these episodes {set(subset_df.idx)}...")
        print(f"Out of these episodes {set(selected_df.idx)}...")
    elif args.viz_random_episodes:
        subset_df = selected_df.sample(args.viz_episodes)
        print(f"Visualizing random episodes... {set(subset_df.idx)}")
    else:
        subset_df = selected_df.groupby(['dataset', 'outcome']).head(args.viz_episodes)
    for episode_idx, row in subset_df.iterrows(): # each row is a trial/episode
        ep_activity = log_analysis.get_activity(row['log'], 
            is_recurrent, 
            do_plot=False)
        traj_df = log_analysis.get_traj_df_tmp(row['log'], # use the tmp version before rerunning eval where unnormalized values are saved to infos. 
                    extended_metadata=False, 
                    squash_action=squash_action)
        traj_df['ep_idx'] = row['idx']
        dataset = row['dataset']
        outcome = row['outcome']
        fprefix = f'{row["dataset"]}_{outcome}'
        OUTPREFIX = smart_outprefix(args.model_dir, dataset, args.out_reldir)

        try:
            # Need to regenerate since no guarantee present already?
            zoom = 1 if 'constant' in dataset else 2    
            zoom = 4 if ('constant' in dataset) and ('HOME' not in outcome) else zoom 
            # zoom = 0 
            zoom = 3 if args.walking else zoom
            agent_analysis.visualize_episodes(episode_logs=[row['log']], 
                                              traj_df=traj_df,
                                              episode_idxs=[row['idx']],
                                              zoom=zoom, 
                                              dataset=row['dataset'],
                                              animate=True,
                                              fprefix=fprefix,
                                              diffusionx=args.diffusionx,
                                              outprefix=OUTPREFIX,
                                              title_text=False, # not supported anyway
                                              legend=False,
                                              invert_colors=args.invert_colors,
                                              image_type=args.image_type,
                                             )    
            if args.animate:
                if args.viz_sensory_angles:
                    agent_analysis.animate_visual_feedback_angles_1episode(traj_df, OUTPREFIX, fprefix, row['idx'])
                if args.viz_neural_activity:
                    log_analysis.animate_activity_1episode(ep_activity, 
                            traj_df, 
                            row['idx'], 
                            fprefix=fprefix,
                            outprefix=OUTPREFIX,
                            pca_dims=3,
                            pca_common=pca_common,
                            invert_colors=args.invert_colors,
                            title=False)
            if args.viz_eigen_values:
                # eig animations/plots
                eig_df = archu.get_eig_df_episode(net, row['log'])
                fname_suffix = f"{fprefix}_ep{row['idx']}"
                if args.animate:
                    archu.animate_Jh_episode(eig_df, 
                        fname_suffix=fname_suffix, 
                        outprefix=OUTPREFIX)
                    eig_vals, eig_vecs = np.linalg.eig(J0)
                archu.plot_eigvec_projections(eig_vals, 
                    eig_vecs, 
                    ep_activity, 
                    fname_suffix=fname_suffix, 
                    outprefix=OUTPREFIX)
            if args.viz_wind_reg:
                agent_analysis.animate_prediction_error_1episode(reg, 'wind_angle_ground_theta', ep_activity, traj_df, OUTPREFIX, fprefix, row['idx'])


        except Exception as e:
            print(f"Exception: {e}", traceback.print_exc())


    # DIRTY Hack to add sparse videos
    # logfiles = natsorted(glob.glob(args.model_dir + '*.pkl')) # never used. Commented out
    for birthx in args.birthxs:
        if birthx == 1 or not birthx or birthx == 'None': # skip. 1 is not sparse. The rest is for supporting old format
            continue 
        sparse_dataset = [str(ds) + f"_{birthx}" for ds in args.use_datasets] 
        
        try:
            sparse_selected_df = log_analysis.get_selected_df(args.model_dir, 
                                sparse_dataset, 
                                n_episodes_home=60, 
                                n_episodes_other=60,
                                min_ep_steps=0)
        except Exception as e:
            print(f"Exception: {e}", traceback.print_exc())
            continue

        sparse_subset_df = sparse_selected_df.groupby(['dataset', 'outcome']).sample(args.viz_episodes)
        if args.viz_wind_reg:
            # fit a linear regression that predicts wind angle from neural activity
            reg = agent_analysis.fit_regression_from_neural_activity_to_latent(sparse_subset_df, \
                'wind_angle_ground_theta') # have not tested if works on sparse cases with different files names. This function should be able to handle that. Just a note so im not surprised. You are doing good.)

        for episode_idx, row in sparse_subset_df.iterrows():
            ep_activity = log_analysis.get_activity(row['log'], 
                is_recurrent, 
                do_plot=False)
            traj_df = log_analysis.get_traj_df_tmp(row['log'], 
                        extended_metadata=False, 
                        squash_action=squash_action)
            
            dataset = "_".join(row['dataset'].split('_')[:-1])
            outcome = row['outcome']
            
            fprefix = f'{row["dataset"]}_{outcome}'
            print("dataset",dataset)
            OUTPREFIX = smart_outprefix(args.model_dir, row['dataset'], args.out_reldir) # row['dataset']: dataset_sparsity

            try:
                # Need to regenerate since no guarantee present already?
                zoom = 1 if 'constant' in dataset else 2    
                zoom = 4 if ('constant' in dataset) and ('HOME' not in outcome) else zoom 
                # zoom = 0 
                zoom = 3 if args.walking else zoom
                agent_analysis.visualize_episodes(episode_logs=[row['log']], 
                                                    # traj_df=traj_df,
                                                    episode_idxs=[row['idx']],
                                                    zoom=zoom, 
                                                    dataset=dataset,
                                                    animate=True,
                                                    fprefix=fprefix,
                                                    outprefix=OUTPREFIX,
                                                    birthx=float(birthx),
                                                    diffusionx=args.diffusionx,
                                                    image_type=args.image_type,
                                                    title_text=False, # not supported anyway
                                                    legend=False,
                                                    invert_colors=args.invert_colors,
                                                    )    
                if args.viz_sensory_angles:
                    agent_analysis.animate_visual_feedback_angles_1episode(traj_df, OUTPREFIX, fprefix, row['idx'])
                if args.viz_neural_activity:
                    log_analysis.animate_activity_1episode(ep_activity, 
                            traj_df, 
                            row['idx'], 
                            fprefix=fprefix,
                            outprefix=OUTPREFIX,
                            pca_dims=3,
                            pca_common=pca_common)
                if args.viz_eigen_values:
                    # eig animations/plots
                    eig_df = archu.get_eig_df_episode(net, row['log'])
                    fname_suffix = f"{fprefix}_ep{row['idx']}"
                    archu.animate_Jh_episode(eig_df, 
                        fname_suffix=fname_suffix, 
                        outprefix=OUTPREFIX)
                    eig_vals, eig_vecs = np.linalg.eig(J0)
                    archu.plot_eigvec_projections(eig_vals, 
                        eig_vecs, 
                        ep_activity, 
                        fname_suffix=fname_suffix, 
                        outprefix=OUTPREFIX)
                if args.viz_wind_reg:
                    agent_analysis.animate_prediction_error_1episode(reg, 'wind_angle_ground_theta', ep_activity, traj_df, OUTPREFIX, fprefix, row['idx'])

            except Exception as e:
                print(f"Exception: {e}", traceback.print_exc())


### MAIN ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Common neural subspace plots/animations')
    parser.add_argument('--model_dir', default=None)
    parser.add_argument('--viz_episodes', type=int, default=2, help='Number of episodes to visualize, selecting from .head()')
    parser.add_argument('--viz_only_these_episodes', type=int, nargs='+', default=False)
    parser.add_argument('--viz_random_episodes', type=bool, default=False)
    parser.add_argument('--walking', type=bool, default=False)
    parser.add_argument('--birthxs', type=float, nargs='+', default=[None])
    parser.add_argument('--diffusionx',  type=float, default=1.0)
    parser.add_argument('--out_reldir', type=str, default='2_videos')
    parser.add_argument('--image_type', type=str, default='png')
    parser.add_argument('--animate', type=bool, default=True)
    parser.add_argument('--invert_colors', type=bool, default=False, help="Make plots with inverted colors - BW")
    parser.add_argument('--viz_wind_reg', type=bool, default=False, help='Visualize wind regression. Fit a line from neural activity to wind direction, and animate the prediction error')
    parser.add_argument('--viz_sensory_angles', type=bool, default=False, help='Visualize sensory angles - only head direction and drift direction at the moment')
    parser.add_argument('--viz_neural_activity', type=bool, default=False, help='Visualize neural population activity of the RNN')
    parser.add_argument('--viz_eigen_values', type=bool, default=False, help='Visualize the eigen spectrum of the RNN population activity')
    parser.add_argument('--use_datasets', type=str,  nargs='+', 
                        default=['constantx5b5', 'switch45x5b5', 'noisy3x5b5'])

    args = parser.parse_args()
    print(args)

    post_eval(args)