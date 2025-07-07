import time
import numpy as np
import itertools
import torch
import pandas as pd
import os
from collections import deque
from data_util import RolloutStorage
# from tamagotchi.eval import eval_lite
import data_util as utils
from env import get_vec_normalize
import matplotlib.pyplot as plt
import mlflow

def get_index_by_dataset(envs, dataset):
    """Get indices of the remotes that loaded the provided dataset."""
    idx = []
    for k,v in envs.remote_directory.items():
        if v['dataset'] == dataset:
            idx.append(k)
    return idx

        
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
            elif '_diff_max' in k:
                # k format: '{ds}_diff_max'
                ds_name = k.split('_diff_max')[0] # get the dataset name
                idx = get_index_by_dataset(envs, ds_name) 
                if len(idx) == 0:
                    print(f"No remote found for {ds_name} lesson")
                else:
                    envs.env_method_at(idx, "update_env_param", {'diff_max': _schedule_dict[curr_step]})
                    print(f"update_env_param 'diff_max': {_schedule_dict[curr_step]} at {curr_step} for remote {idx}")
            elif '_diff_min' in k:
                # k format: '{ds}_diff_min'
                ds_name = k.split('_diff_min')[0] # get the dataset name
                idx = get_index_by_dataset(envs, ds_name) 
                if len(idx) == 0:
                    print(f"No remote found for {ds_name} lesson")
                else:
                    envs.env_method_at(idx, "update_env_param", {'diff_min': _schedule_dict[curr_step]})
                    print(f"update_env_param 'diff_min': {_schedule_dict[curr_step]} at {curr_step} for remote {idx}")
            elif '_rotate_by' in k:
                # k format: '{ds}_rotate_by'
                ds_name = k.split('_rotate_by')[0]
                idx = get_index_by_dataset(envs, ds_name)
                if len(idx) == 0:
                    print(f"No remote found for {ds_name} rotate_by lesson")
                else:
                    envs.env_method_at(idx, "update_env_param", {'rotate_angles': _schedule_dict[curr_step]})
                    print(f"update_env_param 'rotate_angles': {_schedule_dict[curr_step]} at {curr_step} for remote {idx}")
                # probably need to reset the environment since rotate_by gets sampled from full range at init
            else:
                raise NotImplementedError
    return updated # return the course that is updated, if any

def build_tc_schedule_dict(args, total_number_periods, interleave=True, **kwargs):
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
        restart_period: the number of updates between each stage of curriculum. Used for cosine annealing of the learning rate.
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
                print(f"Updated {course} {k} = {course_dirctory[course][k]} to {now_kwargs[k]}", flush=True)
                course_dirctory[course][k] = now_kwargs[k]
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
        if now_course['num_classes'] == 0:
            scheduled_value = [now_course['difficulty_range'][-1]]

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

    # print("[DEBUG] schedule_dict:", schedule_dict)

    # Calculate how often to toggle cosine annealing of the learning rate - interval of curriculum updates
    total_num_updates = len(course_schedule)  # already computed by your code
    restart_period = total_number_periods // (total_num_updates + 1)
    
    # add diff_max sub-lessons that is locked to the wind condition lessons
    lesson_times = [] # all lesson times in the schedule
    for k, v in schedule_dict.items():
        lesson_times.append(list(v.keys()))
    # flatten the list of lists
    lesson_times = list(itertools.chain.from_iterable(lesson_times))
    # sort and remove duplicates
    lesson_times = sorted(set(lesson_times))
    # get the wind condition lessons
    wind_lesson_at = sorted(list(schedule_dict['wind_cond'].keys()))

    available_datasets = args.dataset
    # get a length of period where loc algo grows 
    # get diff_min diff_max of datasets which controls the location algorithm
    # gradually increase diff_max over 4 lessons
    if 'linear' in args.loc_algo:
        # for each dataset find when their lessons are introduced and when the next one is introduced
        total_lesson_time = wind_lesson_at[1] - wind_lesson_at[0] # over this period of time, grow diff_max
        num_lessons = 4
        lesson_time = round(total_lesson_time / num_lessons)
        diff_max_step = (np.array(args.diff_max) - np.array(args.diff_min)) / num_lessons # the step size for the diff_max
        for i, dataset in enumerate(available_datasets):
            # init lesson for each dataset
            lesson_name = f'{dataset}_diff_max'
            if lesson_name not in schedule_dict:
                schedule_dict[lesson_name] = {}
            # get the start time of the lesson
            ds_start = wind_lesson_at[i] # start after the i-th wind cond. is introduced
            # add the lesson to the schedule
            for j in range(4):
                lesson_time_idx = ds_start + j * lesson_time
                step = (j + 1) * diff_max_step[i]
                schedule_dict[lesson_name][lesson_time_idx] = args.diff_min[i] + step
                # eg: {'poisson_mag_narrow_noisy3x5b5_diff_max': {665: 0.4, 720: 0.5, 775: 0.6000000000000001, 830: 0.7000000000000001}
            lesson_name = f'{dataset}_diff_min'
            if lesson_name not in schedule_dict:
                schedule_dict[lesson_name] = {}
            for j in range(4):
                lesson_time_idx = ds_start + j * lesson_time
                step = (j + 1) * diff_max_step[i]
                step = step / 3
                new_diff_min = args.diff_min[i] + step
                new_diff_min = min(0.4, new_diff_min)  # ensure it doesn't go below 0.4
                schedule_dict[lesson_name][lesson_time_idx] = new_diff_min
    
    if 'rotate_by' in args.r_shaping:
        total_lesson_time = wind_lesson_at[1] - wind_lesson_at[0] # over this period of time, grow diff_max
        num_lessons = 4
        lesson_time = round(total_lesson_time / num_lessons)
        for i, dataset in enumerate(available_datasets):
            lesson_name = f'{dataset}_rotate_by'
            if lesson_name not in schedule_dict:
                schedule_dict[lesson_name] = {}
            for j in range(3):
                if j == 0:
                    schedule_dict[lesson_name][lesson_time_idx] = [0, 180]
                elif j == 1:
                    schedule_dict[lesson_name][lesson_time_idx] = [90, -90]
                elif j == 2:
                    schedule_dict[lesson_name][lesson_time_idx] = [0, 90, 180, -90]

        print(f"Added rotate_by lessons to schedule_dict: {schedule_dict}")
        # TODO look at what I actually threw in.. so that can add method for updating this!
        # TODO add method for updating this in update CL function

    return schedule_dict, restart_period

def log_episode(training_log, j, total_num_steps, start, episode_rewards, episode_puffs, episode_plume_densities, episode_wind_directions, num_updates, use_mlflow=True):
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

    log_entry = {
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
        }
    training_log.append(log_entry)
    
    if use_mlflow:
        for k, v in log_entry.items():
            mlflow.log_metric(k, v, step=j)
        
    return training_log


class TrajectoryStorage:
    """
    Lightweight trajectory storage that tracks trajectories by dataset type and episode outcome.
    Collects 2 HOME and 2 OOB trials for each expected dataset.
    Optimizes memory usage by:
    - Only tracking trajectories while still collecting (stops when target reached)
    - Only storing locations for environments that will be used
    - Automatically stopping collection once 2 HOME + 2 OOB per expected dataset are found
    """
    def __init__(self, num_envs, possible_datasets):
        self.num_envs = num_envs
        self.possible_datasets = possible_datasets
        self.expected_datasets = set()  # Datasets expected in current update
        self.trajectories_per_outcome = 2  # Target number per outcome type (HOME/OOB)
        self.is_full = False  # Flag to indicate if storage is full
        self.reset_update()
        # Track ongoing trajectories for each env
        self.ongoing_trajectories = [[] for _ in range(num_envs)]
    
    def set_expected_datasets(self, expected_datasets):
        """Set which datasets to expect during this update"""
        self.expected_datasets = set(expected_datasets)
        # Reset storage structure for expected datasets
        self.stored_trajectories = {
            dataset: {'HOME': [], 'OOB': []} 
            for dataset in self.expected_datasets
        }
        self.dataset_counts = {
            dataset: {'HOME': 0, 'OOB': 0} 
            for dataset in self.expected_datasets
        }
        print(f"Expecting datasets: {self.expected_datasets}")
        print(f"Target: {self.trajectories_per_outcome} HOME + {self.trajectories_per_outcome} OOB per dataset")
    
    def reset_update(self, expected_datasets=None):
        """Reset counters and storage for new update"""
        if expected_datasets is not None:
            self.set_expected_datasets(expected_datasets)
        else:
            # If no datasets specified, reset with empty structure
            self.stored_trajectories = {}
            self.dataset_counts = {}
            self.expected_datasets = set()
        self.is_full = False 
        # Start tracking all environments for the new update
        self.ongoing_trajectories = [[] for _ in range(self.num_envs)]
    
    def add_step(self, infos):
        """Add current step locations to ongoing trajectories (only if still collecting)"""
        if self.is_full:
            return  # Stop tracking once we have enough
            
        for i, info in enumerate(infos):
            if 'location' in info and self.ongoing_trajectories[i] is not None:
                # Only add to trajectories we're actively tracking (None means stopped)
                self.ongoing_trajectories[i].append(info['location'].copy())
    
    def check_episode_done(self, infos, done, tracked_ds=''):
        """Check for completed episodes and store if needed. tracked_ds - stop tracking if this dataset is full."""
        if self.is_full:
            return
        for i, (d, info) in enumerate(zip(done, infos)):
            if d and 'dataset' in info and 'done' in info and self.ongoing_trajectories[i] is not None:
                # Get dataset and outcome type from episode info
                dataset = info['dataset']
                outcome_type = info['done']
                if outcome_type == 'OOT':
                    self.ongoing_trajectories[i] = []
                    continue
                
                # Only store if this dataset is expected and we haven't reached limit for this outcome
                if (dataset in self.expected_datasets and 
                    self.dataset_counts[dataset][outcome_type] < self.trajectories_per_outcome):
                    
                    self.stored_trajectories[dataset][outcome_type].append({
                        'trajectory': self.ongoing_trajectories[i].copy(),
                        'episode_info': info.copy()
                    })
                    self.dataset_counts[dataset][outcome_type] += 1
                if isinstance(tracked_ds, int):
                    tracked_ds = self.possible_datasets[tracked_ds-1] # get the dataset name from the index that is 1-based
                self.is_storage_full(tracked_ds)
                # Decide whether to continue tracking this environment
                if self.is_full:
                    # Stop tracking completely
                    self.ongoing_trajectories[i] = None
                else:
                    # Start tracking new episode
                    self.ongoing_trajectories[i] = []
    
    def get_trajectories(self):
        """Get stored trajectories for current update"""
        return self.stored_trajectories.copy()
    
    def is_storage_full(self, tracked_ds=''):
        if self.is_full:
            return True
        """Check if we've collected enough trajectories for all expected datasets"""
        if not self.expected_datasets:
            return False

        if tracked_ds:  # If a specific dataset is requested, check only that one
            HOME_count = self.dataset_counts[tracked_ds]['HOME']
            OOB_count = self.dataset_counts[tracked_ds]['OOB']
            if (HOME_count < self.trajectories_per_outcome or 
                OOB_count < self.trajectories_per_outcome):
                return False
        else:  # Check if all datasets have reached their targets for both HOME and OOB
            for dataset in self.expected_datasets:
                HOME_count = self.dataset_counts[dataset]['HOME']
                OOB_count = self.dataset_counts[dataset]['OOB']
                if (HOME_count < self.trajectories_per_outcome or 
                    OOB_count < self.trajectories_per_outcome):
                    return False
                
        self.is_full = True  # Set flag if all datasets are full
        return True
    
    def get_collection_status(self):
        """Get current collection status"""
        active_envs = sum(1 for traj in self.ongoing_trajectories if traj is not None)
        status = {
            'expected_datasets': list(self.expected_datasets),
            'dataset_counts': self.dataset_counts.copy(),
            'active_trajectories': active_envs,
            'collection_complete': self.is_full
        }
        
        # Add progress per dataset and outcome type
        for dataset in self.expected_datasets:
            HOME_collected = self.dataset_counts[dataset]['HOME']
            OOB_collected = self.dataset_counts[dataset]['OOB']
            status[f'{dataset}_HOME_progress'] = f"{HOME_collected}/{self.trajectories_per_outcome}"
            status[f'{dataset}_OOB_progress'] = f"{OOB_collected}/{self.trajectories_per_outcome}"
        
        return status
    
    def get_summary_counts(self):
        """Get a concise summary of collection counts"""
        summary = {}
        for dataset in self.expected_datasets:
            HOME_count = self.dataset_counts[dataset]['HOME']
            OOB_count = self.dataset_counts[dataset]['OOB']
            total_count = HOME_count + OOB_count
            target_total = self.trajectories_per_outcome * 2
            summary[dataset] = f"{total_count}/{target_total} (H:{HOME_count}, O:{OOB_count})"
        return summary


def plot_trajectories(traj_storage, envs, save_path="/src/tamagotchi/debug_plot.png", title=None, figsize=None):
    """
    Plot collected trajectories in a grid layout.
    Rows = datasets, Columns = individual trajectories
    Each subplot shows one trajectory with environmental context.
    """
    trajectories = traj_storage.get_trajectories()
    
    if not trajectories:
        print("No trajectories to plot")
        return
    
    # Flatten trajectories by dataset and count max trajectories per dataset
    dataset_trajs = {}
    max_trajs = 0
    
    for dataset, outcomes in trajectories.items():
        # Combine HOME and OOB trajectories for this dataset
        all_trajs = []
        
        # Add HOME trajectories
        for episode in outcomes['HOME']:
            episode_data = episode.copy()
            episode_data['outcome'] = 'HOME'
            all_trajs.append(episode_data)
        
        # Add OOB trajectories  
        for episode in outcomes['OOB']:
            episode_data = episode.copy()
            episode_data['outcome'] = 'OOB'
            all_trajs.append(episode_data)
        
        dataset_trajs[dataset] = all_trajs
        max_trajs = max(max_trajs, len(all_trajs))
    
    if max_trajs == 0:
        print("No trajectories to plot")
        return
    
    # Create subplot grid: rows = datasets, cols = individual trajectories
    datasets = list(dataset_trajs.keys())
    n_rows = len(datasets)
    n_cols = max_trajs
    
    # Set figure size
    if figsize is None:
        figsize = (4 * n_cols, 5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle edge cases for subplot indexing
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes] if n_cols > 1 else [[axes]]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    # Plot each trajectory
    for row, dataset in enumerate(datasets):
        trajs = dataset_trajs[dataset]
        
        for col in range(n_cols):
            ax = axes[row][col]
            
            if col < len(trajs):
                # Plot this trajectory
                episode = trajs[col]
                trajectory = episode['trajectory']
                outcome = episode['outcome']
                color = 'red' if outcome == 'OOB' else 'green'
                
                # Set subplot title
                traj_num = col + 1
                ax.set_title(f"{dataset}\nTraj {traj_num} ({outcome.upper()})", 
                            fontsize=9, pad=10)
                
                if len(trajectory) == 0:
                    ax.text(0.5, 0.5, 'Empty\ntrajectory', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, alpha=0.6)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    continue
                
                # Load and prepare dataset for environmental context
                get_ds_at = get_index_by_dataset(envs, episode['episode_info']['dataset'])[0] 
                data_puffs = envs.get_attr_at(get_ds_at, 'data_puffs_all')[0]
                data_wind = envs.get_attr_at(get_ds_at, 'data_wind_all')[0]
                
                # Apply episode-specific transformations
                drop_idxs = data_puffs['puff_number'].unique()
                drop_idxs = pd.Series(drop_idxs).sample(frac=(1 - episode['episode_info']['plume_density']))
                data_puffs = data_puffs.query("puff_number not in @drop_idxs")
                data_puffs = data_puffs[data_puffs['time'] == episode['episode_info']['t_val']]
                data_puffs = utils.rotate_puffs_optimized(data_puffs, episode['episode_info']['rotate_by'], False)
                data_wind = utils.rotate_wind_optimized(data_wind, episode['episode_info']['rotate_by'], False)

                # Plot puffs and wind vectors
                fig, ax = utils.plot_puffs_and_wind_vectors(data_puffs, data_wind, 
                                                          episode['episode_info']['t_val'], 
                                                          ax=ax, fig=fig, fname='', show=False)
                
                # Plot the trajectory
                x, y = zip(*trajectory)
                ax.plot(x, y, color=color, linewidth=2.5, alpha=0.8)
                
                # Mark start and end points
                ax.plot(x[0], y[0], 'o', color=color, markersize=8, alpha=0.9, 
                       markeredgecolor='black', markeredgewidth=1)  # start
                ax.plot(x[-1], y[-1], 's', color=color, markersize=8, alpha=0.9,
                       markeredgecolor='black', markeredgewidth=1)  # end
                
                # Add trajectory info as text
                traj_length = len(trajectory)
                ax.text(0.02, 0.98, f'Length: {traj_length}', transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
            else:
                # Empty subplot for datasets with fewer trajectories
                ax.set_title(f"{dataset}\n(No trajectory)", fontsize=9, pad=10, alpha=0.6)
                ax.text(0.5, 0.5, 'No trajectory\ncollected', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.4)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # Set axis labels (only on edges to reduce clutter)
            if row == n_rows - 1:  # bottom row
                ax.set_xlabel('X position', fontsize=8)
            if col == 0:  # left column
                ax.set_ylabel('Y position', fontsize=8)
            
            # Make axis equal for proper visualization
            ax.set_aspect('equal', adjustable='box')
            
            # Reduce tick label size
            ax.tick_params(labelsize=7)
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    # Adjust layout to make room for suptitle and text
    if title:
        plt.subplots_adjust(top=0.90)
    plt.subplots_adjust(bottom=0.12, left=0.08, right=0.92)
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Trajectories plot saved to {save_path}")


def plot_update_trajectories(traj_storage, envs, update_num):
    """Convenience function to plot trajectories for a specific update"""
    title = f"Individual Trajectories - Update {update_num}"
    save_path = f"/src/tamagotchi/trajectories_update_{update_num}.png"
    plot_trajectories(traj_storage, envs, save_path=save_path, title=title)
    

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
    
    # See if a chkpt was loaded
    training_log = training_log if training_log is not None else [] 
    last_chkpt_update = len(training_log) # the number of updates already done in case of checkpointing
    eval_log = eval_log if eval_log is not None else []
    # track trajectories for plotting
    best_mean = 0.0
    traj_storage = TrajectoryStorage(num_envs=envs.num_envs, possible_datasets=args.dataset)
    # track stats for logging
    episode_rewards = deque(maxlen=50) 
    episode_plume_densities = deque(maxlen=50)
    episode_puffs = deque(maxlen=50)
    episode_wind_directions = deque(maxlen=50)
    
    # initialize the curriculum schedule
    if args.birthx_linear_tc_steps >= 0: 
        schedule, restart_period = build_tc_schedule_dict(args, num_updates, birthx={'num_classes': args.birthx_linear_tc_steps, 
                                                                'difficulty_range': [0.7, args.birthx], 
                                                                'dtype': 'float', 'step_type': 'linear'}, 
                                          wind_cond={'num_classes': len(args.dataset) - 1, 'difficulty_range': [1, len(args.dataset)], 
                                                     'dtype': 'int', 'step_type': 'linear'}) # wind_cond: in the sequence of args.dataset - first is 1, last is 3
        update_by_schedule(envs, schedule, 0) # update the initialized envs according to the curriculum schedule. The default init values are incorrect, hence this update s.t. reset() returns correctly.
        if not args.dryrun:
            utils.save_tc_schedule(schedule, num_updates, args.num_processes, args.num_steps, args.save_dir)

    
    # finetuning
    # TODO see if works
    if 'finetune' in args.r_shaping:
        # Reset observation normalization stats and reward normalization stats
        envs.reset_obs_norm()
        envs.reset_ret_norm()    
        # warmup the environment with random actions
        # this is to ensure that the observation normalization stats are updated before training
        obs = envs.reset()
        for _ in range(10000):
            action = envs.action_space.sample()  # or use pretrained policy
            obs, reward, done, info = envs.step(action)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs) # https://discuss.pytorch.org/t/which-copy-is-better/56393
    rollouts.to(device)
    start = time.time()
    
    # at each bout of update
    update_range = range(num_updates)
    if last_chkpt_update:
        # if checkpointing, start updating from the last update
        update_range = range(last_chkpt_update, num_updates)

    for j in update_range:
        # decrease learning rate linearly
        if args.use_linear_lr_decay:
            if 'cosine' in args.r_shaping: # HACK! 
                # cosine annealing for the reward shaping
                utils.update_cosine_restart_schedule(agent.optimizer, j, args.lr, restart_period=restart_period)
            else:
                # linear annealing for the learning rate
                utils.update_linear_schedule(agent.optimizer, j, num_updates, args.lr)
        
        ##############################################################################################################
        # Curriculum Learning - update envs according to the curriculum schedule
        ##############################################################################################################
        if args.birthx_linear_tc_steps:
            if last_chkpt_update:
                # update is not 0 when resuming training - need to catch up on the schedule
                for pre_update in range(last_chkpt_update):
                    updated = update_by_schedule(envs, schedule, pre_update)
                obs = envs.reset_after_checkpoint() # reset
                obs = torch.tensor(obs, dtype=torch.float32)
                rollouts.obs[0].copy_(obs) # https://discuss.pytorch.org/t/which-copy-is-better/56393
                rollouts.to(device)
                last_chkpt_update = 0 # reset the last_chkpt_update to 0 after catching up
            updated_var = update_by_schedule(envs, schedule, j)
            
            ##############################################################################################################
            # Checkpointing
            ##############################################################################################################
            if updated_var and not args.dryrun: # save model if an update to env occurred during this trial
                lesson_fpath = os.path.join(args.save_dir, 'chkpt', args.model_fname.replace(".pt", f'_before_{updated_var}{schedule[updated_var][j]}_update{j}.pt'))
                torch.save([
                    actor_critic,
                    getattr(get_vec_normalize(envs), 'obs_rms', None),
                    agent.optimizer.state_dict(),
                ], lesson_fpath)
                # also save the VecNormalize state 
                vecNormalize_state_fname = ''
                if args.if_vec_norm:
                    vecNormalize_state_fname = lesson_fpath.replace(".pt", "_vecNormalize.pkl")
                    envs.venv.save(vecNormalize_state_fname)
                print('Saved', lesson_fpath, vecNormalize_state_fname)

        # Initialize df to track episode statistics
        update_episodes_df = pd.DataFrame(columns=[
            'episode_id', 'dataset', 'outcome', 'reward', 'plume_density', 
            'start_tidx', 'end_tidx', 'location_initial', 'init_angle'
        ]) # track stats of episodes
        episode_counter = 0
        # do this every 10th update
        if j % 10 == 0:
            traj_storage.reset_update(expected_datasets = args.dataset[0:int(envs.wind_directions)]) # track few trajectories for plotting
        ##############################################################################################################
        # at each step of training 
        ##############################################################################################################
        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, activities = actor_critic.act(
                    rollouts.obs[step], 
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            obs, reward, done, infos = envs.step(action)
            if j % 10 == 0:
                traj_storage.add_step(infos)
                traj_storage.check_episode_done(infos, done, tracked_ds=int(envs.wind_directions)) 
            for i, d in enumerate(done): # if done, log the episode info. cCare about what kind of env is encountered
                if d:
                    try:
                        # Note: only ouput these to infos when done
                        episode_counter += 1
                        episode_rewards.append(infos[i]['episode']['r'])
                        episode_plume_densities.append(infos[i]['plume_density']) # plume_density and num_puffs are expected to be similar across different agents. Tracking to confirm. 
                        episode_puffs.append(infos[i]['num_puffs'])
                        episode_wind_directions.append(envs.ds2wind(infos[i]['dataset'])) # density and dataset are logged to see eps. statistics implemented by the curriculum
                        update_episodes_df = utils.update_eps_info(update_episodes_df, infos, episode_counter)
                    except KeyError:
                        raise KeyError("Logging info not found... check why it's not here when done")
                            
            # If done then clean the history of observations in the recurrent_hidden_states. Done in the MLPBase forward method
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            # TODO unsure what this is - only relevant when using TimeLimit wrapper
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks) # ~0.0006s
        ##############################################################################################################
        # UPDATE AGENT 
        ##############################################################################################################
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy, clip_fraction = agent.update(rollouts)
        
        # After update, get stored trajectories
        if j % 10 == 0:
        #     update_trajectories = traj_storage.get_trajectories()
        #     status = traj_storage.get_collection_status()
        #     summary = traj_storage.get_summary_counts()
            plt_path = f"{args.save_dir}/tmp/{args.model_fname.replace('.pt', '_')}trajectories_update{j}.png"
            plot_trajectories(traj_storage, envs, save_path=plt_path)
            if args.mlflow:
                try:
                    mlflow.log_artifact(plt_path, artifact_path=f"figs")
                    os.remove(plt_path)
                except Exception as e:
                    print(f"Error logging artifact {plt_path}: {e}")
                
        utils.log_agent_learning(j, value_loss, action_loss, dist_entropy, clip_fraction, agent.optimizer.param_groups[0]['lr'], use_mlflow=args.mlflow)
        utils.log_eps_artifacts(j, args, update_episodes_df, use_mlflow=args.mlflow)
                
        rollouts.after_update()
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        ##############################################################################################################
        # save for every interval-th episode or for the last epoch
        ##############################################################################################################
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "" and not args.dryrun:

            torch.save([
                actor_critic,
                getattr(get_vec_normalize(envs), 'obs_rms', None),
                agent.optimizer.state_dict(),
            ], args.model_fpath)
            print('Saved', args.model_fpath)
            
            # save the VecNormalize state for evaluation
            if args.if_vec_norm:
                vecNormalize_state_fname = args.model_fpath.replace(".pt", "_vecNormalize.pkl")
                envs.venv.save(vecNormalize_state_fname)

            current_mean = np.median(episode_rewards)
            if current_mean >= best_mean:
                best_mean = current_mean
                fname = f'{args.model_fpath}.best'
                torch.save([
                    actor_critic,
                    getattr(get_vec_normalize(envs), 'obs_rms', None)
                ], fname)
                print('Saved', fname)

        if j % args.log_interval == 0 and len(episode_rewards) > 1 and not args.dryrun:
            training_log = log_episode(training_log, j, total_num_steps, start, episode_rewards, episode_puffs, episode_plume_densities, episode_wind_directions, num_updates)
            # Save training curve
            pd.DataFrame(training_log).to_csv(args.training_log)
    
    # save the final model to mlflow
    if args.mlflow:
        mlflow.log_artifact(args.model_fpath, artifact_path="weights")
        mlflow.log_artifact(args.training_log, artifact_path="training_logs")
        if args.if_vec_norm:
            mlflow.log_artifact(vecNormalize_state_fname, artifact_path="weights")
            # save the final training log
        
    return training_log, eval_log

