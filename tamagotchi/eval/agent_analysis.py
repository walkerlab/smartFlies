import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sklearn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tamagotchi.data_util as data_util
import tamagotchi.config as config
import tamagotchi.eval.log_analysis as log_analysis
from moviepy.editor import ImageClip, concatenate_videoclips
from natsort import natsorted
import contextlib
import os
import tqdm
import torch
import pickle

######################################################################################
### Helper functions ###
# Evaluate agent
def evaluate_agent(agent, 
    env, 
    n_steps_max=200, 
    n_episodes=1, 
    verbose=1):
    np.set_printoptions(precision=4)
    # Rollouts
    episode_logs = []
    for episode in tqdm.tqdm(range(n_episodes)): 
        trajectory = []
        observations = []
        actions = []
        rewards = []
        infos = []

        obs = env.reset()
        reward = 0
        done = False
        cumulative_reward = 0
        for step in range(n_steps_max):  
            # Select next action
            action = agent.act(obs, reward, done)
            # print("step, action:", step, action)
            # Step environment w/ action
            # print(action)
            obs, reward, done, info = env.step(action) # obs: [1, 3] 

            trajectory.append( info[0]['location'] )
            observations.append( obs )
            actions.append( action )
            rewards.append( reward )
            infos.append( info )

            cumulative_reward += reward
            if verbose > 1:
                print("{}: Action: {}, Odor:{:.6f}, Wind:({:.2f}, {:.2f}) Reward:{}, Loc:{}, Angle:{}".format(
                    step+1, 
                    action, 
                    obs[0][2], # odor
                    obs[0][0], # wind_x
                    obs[0][1], # wind_y
                    reward, 
                    [ '%.2f' % elem for elem in info[0]['location'] ],
                    '%.2f' % np.rad2deg( np.angle(info[0]['angle'][0] + 1j*info[0]['angle'][1]) )
                ))    
            if done:
                break
        if verbose > 0:
            print("Episode {} stopped at {} steps with cumulative reward {}".format(episode + 1, step + 1, cumulative_reward))

        episode_log = {
            'trajectory': trajectory,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'infos': infos,
        }
        episode_logs.append(episode_log)

    return episode_logs

######################################################################################
## Behavior Analysis ##

# TODO plot wind and wind estimation with this function
def visualize_single_episode(data_puffs, data_wind, traj_df, 
    episode_idx, zoom=1, t_val=None, title_text=None, output_fname=None, 
    show=True, colorby=None, vmin=0, vmax=1, plotsize=None, xlims=None, ylims=None, legend=True,
    invert_colors=False, kwargs={}):
    if 'scatter_size' in kwargs.keys():
        if kwargs['scatter_size'] is None:
            scatter_size = 15
        else:
            scatter_size = kwargs['scatter_size']

    plotsize = (8,8) if plotsize is None else plotsize
    if 'subplot_spec' in kwargs.keys():
        fig = kwargs['figure'] 
        gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(
                        nrows=kwargs['nrows'] if 'nrows' in kwargs.keys() else 1,
                        ncols=kwargs['ncols'] if 'ncols' in kwargs.keys() else 1,
                        subplot_spec=kwargs['subplot_spec'],
                        )
        ax = fig.add_subplot(gs00[:, :])
        ax = fig.add_subplot(kwargs['subplot_spec'])
    else:
        ax = None # do not pass into plot_puffs_and_wind_vectors if not subplot_spec. plot_puffs_and_wind_vectors will create its own figure
        fig = None
    
    aspect_ratio = kwargs['aspect_ratio'] if 'aspect_ratio' in kwargs.keys() else False
        
    try:      
        fig, ax = data_util.plot_puffs_and_wind_vectors(data_puffs, data_wind, t_val, ax = ax, fig=fig,
                                           fname='', plotsize=plotsize, aspect_ratio=aspect_ratio, show=show, invert_colors=invert_colors)
    except Exception as e:
        print(f"[Error] visualize_single_episode: {episode_idx} {e}")
        return None, None

    # Crosshair at source
    if invert_colors:
        ax.plot([0, 0],[-0.3,+0.3],'w-', linestyle = ":", lw=2) # presentation black background, white lines
        ax.plot([-0.3,+0.3],[0, 0],'w-', linestyle = ":", lw=2)
    else:
        ax.plot([0, 0],[-0.3,+0.3],'k-', linestyle = ":", lw=2) # manuscript white background, black lines
        ax.plot([-0.3,+0.3],[0, 0],'k-', linestyle = ":", lw=2)

    # Handle custom colorby
    if colorby is not None and type(colorby) is not str:
        colors = colorby # assumes that colorby is a series
        colorby = 'custom'

    # Line for trajectory
    linecolor='black'
    ax.plot(traj_df.iloc[:,0], traj_df.iloc[:,1], c=linecolor, lw=0.5) # Red line!
    ax.scatter(traj_df.iloc[0,0], traj_df.iloc[0,1], c='black', 
        edgecolor='black', marker='o', s=100) # Start

    # Scatter plot for odor/regime etc.
    # Default: colors indicate odor present/absent
    if colorby is None:
        colors = [config.traj_colormap['off'] if x <= config.env['odor_threshold'] else config.traj_colormap['on'] for x in traj_df['odor_eps_log']]
        cm = plt.cm.get_cmap('winter') # not sure if makes a difference
        ax.scatter(traj_df.iloc[:,0], traj_df.iloc[:,1], 
            c=colors, s=scatter_size, cmap=cm, vmin=vmin, vmax=vmax, alpha=1.0)
    if colorby is not None and colorby == 'complete': 
        # Colors indicate % trajectory complete
        colors = traj_df.index/len(traj_df)
        cm = plt.cm.get_cmap('winter')
        ax.scatter(traj_df.iloc[:,0], traj_df.iloc[:,1], 
            c=colors, s=scatter_size, cmap=cm, vmin=vmin, vmax=vmax, alpha=1.0)
    if colorby is not None and colorby == 'regime': 
        colors = [ config.regime_colormap[x] for x in traj_df['regime'].to_list() ]
        cm = None
        ax.scatter(traj_df.iloc[:,0], traj_df.iloc[:,1], 
            c=colors, s=scatter_size, cmap=cm, vmin=vmin, vmax=vmax, alpha=1.0)
    if colorby is not None and colorby == 'custom': 
        cm = plt.cm.get_cmap('winter')
        ax.scatter(traj_df.iloc[:,0], traj_df.iloc[:,1], 
            c=colors, s=scatter_size, cmap=cm, vmin=vmin, vmax=vmax, alpha=1.0)

    if zoom == 1: # Constant wind
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-1.5, +1.5)
    if zoom == 2: # Switch or Noisy
        ax.set_xlim(-1, 10.5)
        # ax.set_ylim(-5, 5)
        ax.set_ylim(-1.5, 5)
    if zoom == 3: # Walking
        ax.set_xlim(-0.15, 0.5)
        ax.set_ylim(-0.2, 0.2)
    if zoom == 4: # constant + larger arena
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-3, +3)
    if zoom == -1: # Adaptive -- fine for stills, jerky when used for animations
        ax.set_xlim(-0.5, 10.1)
        y_max = max(data_puffs[data_puffs.time == t_val].y.max(), traj_df.iloc[:,1].max()) + 0.5
        y_min = min(data_puffs[data_puffs.time == t_val].y.min(), traj_df.iloc[:,1].min()) - 0.5
        # print('y_max', data_puffs[data_puffs.time == t_val].y.max(), traj_df.loc[:,1].max())
        # print('y_min', data_puffs[data_puffs.time == t_val].y.min(), traj_df.loc[:,1].min())
        ax.set_ylim(y_min, y_max)
    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1])
    if ylims is not None:
        print(ylims)
        ax.set_ylim(ylims[0], ylims[1])

    if zoom > 0:
        ax.set_xlabel('Arena length [m]')
        ax.set_ylabel('Arena width [m]')
    if title_text is not None:
        ax.set_title(title_text)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        patch1 = mpatches.Patch(color=config.traj_colormap['off'], label='Off plume')   
        patch2 = mpatches.Patch(color=config.traj_colormap['on'], label='On plume')   
        handles.extend([patch1, patch2])
        # plt.legend(handles=handles, loc='upper left')
        # plt.legend(handles=handles, loc='lower right')
        if 'fontsize' in kwargs.keys():
            ax.legend(handles=handles, loc='upper right', fontsize=kwargs['fontsize']) 
        else:
            ax.legend(handles=handles, loc='upper right')
        # https://stackoverflow.com/questions/12848808/set-legend-symbol-opacity-with-matplotlib
        # for lh in leg.legendHandles: 
            # lh.set_alpha(1) # no longer compatible
    if invert_colors:
        # for Bing presentation... set background to black and text to white
        ax.set_facecolor('black')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        fig.set_facecolor('black')
    if output_fname is not None:
        if output_fname.endswith('.pdf'):
            plt.savefig(output_fname, bbox_inches='tight', format='pdf', dpi=300)
        else:
            plt.savefig(output_fname, bbox_inches='tight')
    return fig, ax


def animate_single_episode(
    data_puffs, data_wind, traj_df, 
    t_vals, t_vals_all,
    episode_idx, outprefix, fprefix, zoom, 
    colorby=None, plotsize=None, legend=True, invert_colors=False, scatter_size=None):
    
    n_tvals = len(t_vals) 
    if n_tvals == 0:
       print("n_tvals == 0!") 
    output_fnames = [] 
    skipped_frames = 0
    if not os.path.exists(f'{outprefix}/tmp/'):
        os.makedirs(f'{outprefix}/tmp/')
        
    t_val_min = None
    for t_idx in tqdm.tqdm(range(n_tvals)):
        traj_df_subset = traj_df.iloc[:t_idx+1,:] # feed trajectory incrementally 
        t_val = t_vals[t_idx]
        if t_val_min is None:
            t_val_min = t_val
        if t_val not in t_vals_all: # TODO: HACK to skip when t_val missing in puff_data!!
            skipped_frames += 1
            continue
        output_fname = f'{outprefix}/tmp/{fprefix}_ep{episode_idx}_step{t_idx:05d}.png'
        output_fnames.append(output_fname)
        title_text = f"episode:{episode_idx} step:{t_idx+1} [t:{t_val:0.2f}]"
        # title_text = f"Step:{t_idx+1} [Time:{t_val:0.2f}]"
        # title_text = f"Time:{t_val:0.2f}s"
        title_text = f"Time:{t_val-t_val_min:0.2f}s"
        fig, ax = visualize_single_episode(data_puffs, data_wind, 
            traj_df_subset, 
            episode_idx, 
            zoom, 
            t_val=t_val, 
            title_text=title_text, # not supported currently
            output_fname=output_fname,
            show=False,
            colorby=None,
            plotsize=plotsize,
            legend=legend,
            invert_colors=invert_colors,
            kwargs={'scatter_size':scatter_size}
            )
        # release memory from matplotlib
        fig.clf()
        ax.cla()
        plt.close()
    if skipped_frames > 0:
        print(f"Skipped {skipped_frames} out of {n_tvals} frames!")
    output_fnames = natsorted(output_fnames,reverse=False)
    if len(output_fnames) == 0:
        print("No valid frames!")
        return

    clips = [ImageClip(f).set_duration(0.08) for f in output_fnames] # 
    concat_clip = concatenate_videoclips(clips, method="compose")
    fanim = f"{outprefix}/{fprefix}_ep{episode_idx:03d}.mp4"
    concat_clip.write_videofile(fanim, fps=15, verbose=False, logger=None)
    # fanim = f"{outprefix}/{fprefix}_ep{episode_idx:03d}.gif"
    # concat_clip.write_gif(fanim, fps=30, verbose=False, logger=None)
    print("Saved", fanim)
    
    for f in output_fnames:
        # https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)


def visualize_episodes(episode_logs, 
                       traj_df=None,
                       zoom=1, 
                       outprefix=None, 
                       title_text=True, 
                       animate=False,
                       fprefix='trajectory',
                       dataset='constant',
                       birthx=1.0,
                       diffusionx=1.0,
                       episode_idxs=None,
                       colorby=None,
                       vmin=0, vmax=1,
                       plotsize=None,
                       legend=True,
                       invert_colors=False,
                       image_type='png',
                       scatter_size=None,
                       fontsize=None,
                       ):

    # Trim/preprocess loaded dataset!
    t_starts = []
    t_ends = []
    for log in episode_logs: 
        t_starts.append( log['infos'][0][0]['t_val'] )
        t_ends.append( log['infos'][-1][0]['t_val'] )
    # try:
    #     radiusx = episode_logs[-1]['infos'][0][0]['radiusx']
    # except Exception as e:
    radiusx = 1.0    
    data_puffs_all, data_wind_all = data_util.load_plume(dataset, 
        t_val_min=min(t_starts)-1.0, 
        t_val_max=max(t_ends)+1.0,
        radius_multiplier=radiusx,
        diffusion_multiplier=diffusionx,
        puff_sparsity=np.clip(birthx, a_min=0.01, a_max=1.00),
        )
    t_vals_all = data_puffs_all['time'].unique()

    # check if traj_df is None - get traj_df from episode_logs for each episode
    get_traj_df = True if traj_df is None else False
    
    if not get_traj_df:
        traj_df_all = traj_df.copy()
    
    # Plot and animate individual episodes
    n_episodes = len(episode_logs)
    if episode_idxs is None:
        episode_idxs = [i for i in range(n_episodes)]
    
    figs, axs = [], []
    for episode_idx in range(n_episodes): 
        episode_idx_title = episode_idxs[episode_idx] # Hack to take in custom idxs
        ep_log = episode_logs[episode_idx]
        t_val_end = t_ends[episode_idx]

        if get_traj_df: # get traj_df for each row of epsiode_logs
            traj_df = log_analysis.get_traj_df_tmp(ep_log, 
            extended_metadata=False, 
            squash_action=True)
        else:
            traj_df = traj_df_all.query("ep_idx == @episode_idx_title").copy()

        if title_text:
            title_text = f"ep:{episode_idx_title} t:{t_val_end:0.2f} "
            title_text += "step: {}".format(traj_df.shape[0])
        else:
            title_text = None

        if outprefix is not None:
            output_fname = f"{outprefix}/{fprefix}_{episode_idx_title:03}.png"
            if image_type == 'pdf':
                output_fname = output_fname.replace('png', 'pdf')
            print(f"visualize_episodes: saving to {output_fname}")
        else:
            output_fname = None

        # Flip plume about x-axis (generalization)
        flipx = ep_log['infos'][0][0]['flipx']
        if flipx < 0:
            data_wind = data_wind_all.copy() # flip this per episode
            data_puffs = data_puffs_all.query("time <= @t_val_end + 1").copy()
            data_wind.loc[:,'wind_y'] *= flipx   
            data_puffs.loc[:,'y'] *= flipx 
        else:
            data_wind = data_wind_all.query("time <= @t_val_end + 1")
            data_puffs = data_puffs_all.query("time <= @t_val_end + 1")

        t_vals = [record[0]['t_val'] for record in ep_log['infos']]
        ylims = xlims = None
        if zoom == 0:
            print("adaptive ylims")
            xlims = [-0.5, 10.1]
            ylims = [ traj_df['loc_y'].min() - 0.25, traj_df['loc_y'].max() + 0.25 ]
        fig, ax = visualize_single_episode(data_puffs, data_wind, 
            traj_df, episode_idx_title, zoom, t_val_end, 
            title_text, output_fname, colorby=colorby,
            vmin=vmin, vmax=vmax, plotsize=plotsize, 
            xlims=xlims, ylims=ylims, legend=legend, invert_colors=invert_colors, kwargs={'scatter_size':scatter_size,'fontsize':fontsize})
        figs.append(fig)
        axs.append(ax)

        if animate:
            animate_single_episode(data_puffs, data_wind, traj_df, 
                t_vals, t_vals_all, episode_idx_title, outprefix, 
                fprefix, zoom, colorby=colorby, plotsize=plotsize, legend=legend, invert_colors=invert_colors)

    return figs, axs

def visualize_episodes_metadata(episode_logs, zoom=1, outprefix=None):
    n_episodes = len(episode_logs)
    for episode_idx in range(n_episodes): 
        # Plot Observations over time
        obs = [ x[0] for x in episode_logs[episode_idx]['observations'] ]
        obs = pd.DataFrame(obs)
        obs.columns = ['wind_x', 'wind_y', 'odor']
        obs['wind_theta'] = obs.apply(lambda row: vec2rad_norm_by_pi(row['wind_x'], row['wind_y']), axis=1)
        # axs = obs.loc[:,['wind_theta','odor']].plot(subplots=True, figsize=(10,4), title='Observations over time')
        # axs[-1].set_xlabel("Timesteps")

        # Plot Actions over time
        act = [ x[0] for x in episode_logs[episode_idx]['actions'] ]
        act = pd.DataFrame(act)
        act.columns = ['step', 'turn']
        # axs = act.plot(subplots=True, figsize=(10,3), title='Actions over time')
        # axs[-1].set_xlabel("Timesteps")

        merged = pd.merge(obs, act, left_index=True, right_index=True)
        axs = merged.loc[:,['wind_theta','odor','step', 'turn']].plot(subplots=True, figsize=(10,5), title='Observations & Actions over time')
        axs[-1].set_xlabel("Timesteps")
        axs[0].set_ylim(-np.pi, np.pi)        
        # axs[1].set_ylim(0, 0.5)        
        # axs[2].set_ylim(0, 1)        
        # axs[3].set_ylim(-1, 1)        
        # plt.tight_layout()

        if outprefix is not None:
            fname = "{}_ep{}_meta.png".format(outprefix, episode_idx)
            plt.savefig(fname)
        plt.close()


#### Behavior Analysis ####
def sample_obs_action_pair(agent, off_plume=False):
    wind_relative_angle_radians = np.random.uniform(low=-np.pi, high=+np.pi)
    # Always in plume
    if off_plume:
        odor_observation = 0.0
    else:
        odor_observation = np.random.uniform(low=0.0, high=0.3) # Appropriate distribution?
    
    wind_observation = [ np.cos(wind_relative_angle_radians), np.sin(wind_relative_angle_radians) ]
    obs = np.array([wind_observation + [odor_observation] ]).astype(np.float32)
    action = agent.act(obs, reward=0, done=False)
    return np.concatenate([obs.flatten(), action.flatten()])
#     return [obs, action]

def get_samples(agent, N=1000, off_plume=False):
    samples = [ sample_obs_action_pair(agent, off_plume) for i in range(N) ]
    return samples


# Add a wind_theta column
def vec2rad_norm_by_pi(x, y):
    return np.angle( x + 1j*y, deg=False )/np.pi # note div by np.pi!

def get_sample_df(agent, N=1000, off_plume=False):
    samples = get_samples(agent, N, off_plume)
    sample_df = pd.DataFrame(samples)
    sample_df.columns = ['wind_x', 'wind_y', 'odor', 'step', 'turn']
    sample_df['wind_theta'] = sample_df.apply(lambda row: vec2rad_norm_by_pi(row['wind_x'], row['wind_y']), axis=1)
    return sample_df

def visualize_policy_from_samples(sample_df, outprefix=None):
    # Plot turning policy
    plt.figure(figsize=(1.1, 1.1))
    plt.scatter(sample_df['wind_theta'], sample_df['turn']-0.5, alpha=0.5, s=3)
    plt.xlabel("Wind angle [$\pi$]")
    plt.ylabel("Turn angle [$\pi$]")
    # plt.yticks([])
    # plt.title("Agent turn policy")
    plt.xlim(-1, +1)    
    plt.ylim(-1, +1)
    if outprefix is not None:
        fname = "{}_policy_turn.png".format(outprefix)
        plt.savefig(fname)
    # plt.close()

# agent_analysis.visualize_episodes_metadata([one_trial], zoom=1, outprefix=None)
# Copy-pasted from agent_analysis.py
def get_obs_act_for_episode(episode, plot=True, stacked=True):
    obs = [ x[0] for x in episode['observations'] ]
    obs = pd.DataFrame(obs)
    if stacked: # Stacked models
        obs = obs.iloc[:, -3:]
    obs.columns = ['wind_x', 'wind_y', 'odor']
    obs['wind_theta'] = obs.apply(lambda row: vec2rad_norm_by_pi(row['wind_x'], row['wind_y']), axis=1)
    act = [ x[0] for x in episode['actions'] ]
    act = pd.DataFrame(act)
    act.columns = ['step', 'turn']
    merged = pd.merge(obs, act, left_index=True, right_index=True)
    return merged

######################################################################################
### Neural activity analysis ###

#######################################################################################
### visualize sensory inputs over a trajectory ###


def animate_visual_feedback_angles_1episode(traj_df, outprefix, fprefix, episode_idx):
    """
    Animates the visual feedback angles for a single episode.

    Args:
        traj_df (DataFrame): The trajectory DataFrame containing the time steps.
        outprefix (str): The output prefix for saving the animation files.
        fprefix (str): The file prefix for naming the animation files.
        episode_idx (int): The index of the episode.

    Returns:
        None
    """
    def animate_visual_feedback_angles_single_frame(df_current_time_step, output_fname):
        """
        Animates the visual feedback angles for a single frame.

        Args:
            df_current_time_step (DataFrame): The DataFrame for the current time step.
            output_fname (str): The output filename for saving the animation.

        Returns:
            None
        """
        allocentric_fname = output_fname.replace('.png', '_allocentric.png')
        egocentric_fname = output_fname.replace('.png', '_egocentric.png')
        # get visual feedback angles
        allo_head_direction_theta = np.angle(df_current_time_step['agent_angle_x'] + 1j*df_current_time_step['agent_angle_y'], deg=False)
        print(f"[NOTE] ego_course_direction_theta is assumed to be flipped by pi for the current implementation. Please check if this is the case for your data. This function is calculating ego_course_direction_theta from scatch.", flush=True)
        ego_course_direction_theta = np.angle(df_current_time_step['ego_course_direction_x'] + 1j*df_current_time_step['ego_course_direction_y'], deg=False) - np.pi # subtract pi because currently the ground velocity calculation is flipped

        # plot unit vector of angles in allocentric frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.quiver(0,0, allo_head_direction_theta, 1, angles='xy', scale_units='xy', scale=1., color='red')
        ax.quiver(0,0, allo_head_direction_theta + ego_course_direction_theta, 1, angles='xy', scale_units='xy', scale=1., color='orange')
        ax.plot(0, 2, color='black', marker='o', markersize=5)
        ax.set_rmax(1)
        ax.set_rticks([])
        plt.title('Allocentric head direction and course direction')
        plt.savefig(allocentric_fname, bbox_inches='tight')
        # release memory from matplotlib
        fig.clf()
        ax.cla()
        plt.close()

        # plot unit vector of angles in egocentric frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.quiver(0,0, 0, 1, angles='xy', scale_units='xy', scale=1., color='red')
        ax.quiver(0,0, ego_course_direction_theta, 1, angles='xy', scale_units='xy', scale=1, color='orange')
        ax.plot(0, 2, color='black', marker='o', markersize=5)
        ax.set_rmax(1)
        ax.set_rticks([])
        plt.title('Egocentric head direction and course direction')
        ax.set_theta_zero_location("N")
        plt.savefig(egocentric_fname, bbox_inches='tight')
        # release memory from matplotlib
        fig.clf()
        ax.cla()
        plt.close()

    output_fnames = [] 
    for t_idx, df_current_time_step in traj_df.iterrows():    
        if not os.path.exists(f'{outprefix}/tmp/'):
            os.makedirs(f'{outprefix}/tmp/')
        output_fname = f'{outprefix}/tmp/{fprefix}_ep{episode_idx}_step{t_idx:05d}.png'
        output_fnames.append(output_fname)

        animate_visual_feedback_angles_single_frame(df_current_time_step, output_fname)
    output_fnames = natsorted(output_fnames, reverse=False)

    for plot_type in ['allocentric', 'egocentric']:
        cur_fnames = [f.replace('.png', f"_{plot_type}.png") for f in output_fnames]
        clips = [ImageClip(f).set_duration(0.08) for f in cur_fnames] 
        concat_clip = concatenate_videoclips(clips, method="compose")
        fanim = f"{outprefix}/{fprefix}_ep{episode_idx:03d}_{plot_type}.mp4"
        concat_clip.write_videofile(fanim, fps=15, verbose=False, logger=None)
        print("Saved", fanim)
        
    for f in output_fnames:
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)


def fit_regression_from_neural_activity_to_latent(eval_log_pkl_df: pd.DataFrame, latent_col_name: str, stacked_neural_activity: np.ndarray = None, stacked_traj_df: pd.DataFrame = None) -> sklearn.linear_model.LinearRegression:
    """
    Fits a linear regression model to predict the latent variable from neural activity.

    Args:
        eval_log_pkl_df (pd.DataFrame): DataFrame containing evaluation logs.
        latent_col_name (str): Name of the column representing the latent variable.
        stacked_neural_activity (np.ndarray, optional): Stacked neural activity data. If not provided, it will be loaded from the evaluation logs.
        stacked_traj_df (pd.DataFrame, optional): Stacked trajectory DataFrame. If not provided, it will be loaded from the evaluation logs.

    Returns:
        sklearn.linear_model.LinearRegression: Fitted linear regression model.
    """
    
    # load and stack data if not provided
    get_neural_activity = get_traj_df = False
    if stacked_neural_activity is None:
        get_neural_activity = True
    if stacked_traj_df is None:
        get_traj_df = True
    if get_neural_activity or get_traj_df:    
        is_recurrent = True
        squash_action = True
        h_episodes = []
        traj_dfs = []
        for idx, row  in tqdm.tqdm(eval_log_pkl_df.iterrows()):
            episode_log = row['log']
            if get_neural_activity:
                ep_neural_activity = log_analysis.get_activity(episode_log, is_recurrent, do_plot=False)
                h_episodes.append(ep_neural_activity)
            if get_traj_df:
                traj_df = log_analysis.get_traj_df_tmp(episode_log, 
                                                extended_metadata=False, 
                                                squash_action=squash_action)
                traj_df['tidx'] = np.arange(traj_df.shape[0], dtype=int)
                for colname in ['dataset', 'idx', 'outcome']:
                    traj_df[colname] = row[colname] 
                traj_dfs.append(traj_df)
        if get_neural_activity:
            stacked_neural_activity = np.vstack(h_episodes)
        if get_traj_df:
            stacked_traj_df = pd.concat(traj_dfs)

    
    # linear regression
    Y = stacked_traj_df[latent_col_name]
    X = stacked_neural_activity[ ~Y.isna() ]
    Y = Y[ ~Y.isna() ]
    reg = sklearn.linear_model.LinearRegression().fit(X, Y)
    print("R2 score:", reg.score(X, Y))
    return reg
    
    
def animate_prediction_error_1episode(reg, latent, ep_activity, traj_df, outprefix, fprefix, episode_idx):
    """
    Animates the visual feedback angles for a single episode.

    Args:
        traj_df (DataFrame): The trajectory DataFrame containing the time steps.
        outprefix (str): The output prefix for saving the animation files.
        fprefix (str): The file prefix for naming the animation files.
        episode_idx (int): The index of the episode.

    Returns:
        None
    """
    def animate_prediction_error_single_frame(now_pred_errors, output_fname, xlim, ylim):
        """
        Animates the prediction error for a single frame.

        Args:
            prediction_errors (np.ndarray): The prediction errors.
            output_fname (str): The output filename for saving the animation.

        Returns:
            None
        """
        fig = plt.figure()
        fig.plot(now_pred_errors, color='black')
        plt.xlim(0, xlim)
        plt.ylim(0, ylim)
        plt.xlabel('Time')
        plt.ylabel('Absolute prediction error')
        # figure size
        plt.gcf().set_size_inches(10, 3)
        # R2 score
        R2 = sklearn.metrics.r2_score(targets, predictions)
        textstr = f"R\u00b2 = {R2}"
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                verticalalignment='top')
        plt.savefig(output_fname, bbox_inches='tight', dpi=300)
        # release memory from matplotlib
        fig.clf()
        plt.close()
    # predict latent
    predictions = reg.predict(ep_activity)
    targets = traj_df[latent]
    # get errors
    prediction_errors = np.abs(predictions - targets)  
    xlim = len(predictions)
    ylim = np.max(prediction_errors)  
    # plot prediction error over time
    output_fnames = []
    for i in range(len(prediction_errors)):
        curr_errors = prediction_errors[:i] 
        if not os.path.exists(f'{outprefix}/tmp/'):
            os.makedirs(f'{outprefix}/tmp/')
        output_fname = f'{outprefix}/tmp/{fprefix}_ep{episode_idx}_{latent}_pred_error_step{i:05d}.png'
        output_fnames.append(output_fname)
        animate_prediction_error_single_frame(curr_errors, output_fname, xlim, ylim)
    output_fnames = natsorted(output_fnames, reverse=False)

    clips = [ImageClip(f).set_duration(0.08) for f in output_fnames] 
    concat_clip = concatenate_videoclips(clips, method="compose")
    fanim = f'{outprefix}/{fprefix}_ep{episode_idx}_{latent}_pred_error.mp4'
    concat_clip.write_videofile(fanim, fps=15, verbose=False, logger=None)
    print("Saved", fanim)
        
    for f in output_fnames:
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)
            
            
#######################################################################################
### code for perturbing along a dimension of interest ###
#######################################################################################

def import_orthogonal_basis(fname):
    # check if file is a numpy file or a pickle file
    if fname.endswith('.npy'):
        # Example file: /src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_ALL_noisy_wind_0.001/eval/plume_951_23354e57874d619687478a539a360146/orthogonal_basis_with_wind_encoding_subspace_951.npy
        # generated from /src/JH_boilerplate/agent_evaluatiion/wind_encoding_perturbation/perturb_along_dim.ipynb
        # expected shape: (64 x64), where each row is a basis vector and the first row is the wind encoding subspace
        ortho_set = np.load(fname)
        # check for orthogonality
        dot_product = np.dot(ortho_set, ortho_set.T)
        assert(np.allclose(dot_product, np.eye(64,64), atol=1e-2)), "The basis vectors failed the orthogonality set - check the loaded file"
        
        return (ortho_set)
    elif fname.endswith('.pkl'):
        # Example file: /src/data/wind_sensing/apparent_wind_visual_feedback/sw_dist_logstep_ALL_noisy_wind_0.001/eval/plume_951_23354e57874d619687478a539a360146/ranked_orthogonal_basis_and_var_with_wind_encoding_subspace_951.pkl
        # generated from /src/JH_boilerplate/agent_evaluatiion/wind_encoding_perturbation/noise_generation.ipynb
        # expected shape: (64 x64), where each row is a basis vector and the first row is the wind encoding subspace
        # expected shape: (64, ), where each value is the NN activity variance along the corresponding basis vector
        with open(fname, 'rb') as f:
            file = pickle.load(f)
            ortho_set = file['PCs']
            variances = file['PC_variances']
        # check for orthogonality
        dot_product = np.dot(ortho_set, ortho_set.T)
        assert(np.allclose(dot_product, np.eye(64,64), atol=1e-2)), "The basis vectors failed the orthogonality set - check the loaded file"
        
        return (ortho_set, variances)


def express_vec_as_sum_of_basis(v, basis):
    # express a vector v as a linear combination of basis vectors
    # returns the coefficients of the linear combination
    # This arithmetic was verified in /src/JH_boilerplate/agent_evaluatiion/wind_encoding_perturbation/perturb_along_dim.ipynb
    coef = np.dot(v, basis.T) / np.dot(basis, basis.T)
    return np.diagonal(coef)


def generate_white_noise(sigma, sample_by='normal'):
    # Sample noise coefficients for scaling the basis vectors. Perturbation = sum_i coef_i * basis_i
    # return shape (len(sigma),)
    # Sample a noise coefficient 
    if sample_by == 'normal':
        return np.random.normal(0, sigma) # N(0, sigma_i)
    elif sample_by == 'uniform':
        return np.random.uniform(-sigma, sigma)
    else:
        raise ValueError("[ERROR] generate_white_noise: sample_by should be 'normal' or 'uniform'")


def perturb_rnn_activity(rnn_activity, ortho_set, sigma, perturb_direction, sample_noise_by='normal', return_perturb_by=False):
    '''
    sigma (float or matrix): standard deviation of the noise
    '''
    # perturb the rnn activity
    noise_constant = generate_white_noise(sigma, sample_by=sample_noise_by)
    if perturb_direction == 'subspace':
        # express the noise as a linear combination of the basis vectors
        perturb_by = noise_constant[0] * ortho_set[0] # first row is the wind encoding subspace
    elif perturb_direction == 'all':
        perturb_by = noise_constant * ortho_set
    elif perturb_direction == 'nullspace':
        # express the noise as a linear combination of the basis vectors
        perturb_by = noise_constant[1:] @ ortho_set[1:] # first row is the wind encoding subspace, so exclude it
    elif perturb_direction == 'subspace_WN_in_nullspace':
        # sample null direction by their relative variance and perturb in that direction by the noise_constant drawn from the wind variance
        null_dir_sample = noise_constant[1:] @ ortho_set[1:] # first row is the wind encoding subspace, so exclude it
        # get the sample as an unit vector
        u_null_dir_sample = null_dir_sample / np.linalg.norm(null_dir_sample) 
        # perturb in the direction of the nullspace with the WN drawn from wind variance. This ensures 1. perturb by the same strength as wind subspace, 2. the nullspace direction reflects their variabiliy
        perturb_by = noise_constant[0] * u_null_dir_sample
    else:
        raise ValueError("perturb_direction should be 'subspace', 'all', 'nullspace', or 'subspace_WN_in_nullspace'")
    perturb_by = torch.from_numpy(perturb_by)
    perturb_by = perturb_by.to(device=rnn_activity.device.type, dtype=torch.float32)
    if return_perturb_by:
        return rnn_activity + perturb_by, perturb_by
    return rnn_activity + perturb_by
