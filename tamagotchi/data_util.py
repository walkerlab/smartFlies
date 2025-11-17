from tamagotchi import config
from tamagotchi import sim_utils as sim_analysis
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import mlflow
import math

def load_plume(
    dataset='constant', 
    t_val_min=None,
    t_val_max=None,
    env_dt=0.04,
    puff_sparsity=1.00,
    radius_multiplier=1.00,
    diffusion_multiplier=1.00,
    data_dir=config.datadir,
    ):
    print("[load_plume]",dataset)
    puff_filename = f'{data_dir}/puff_data_{dataset}.pickle' 
    wind_filename = f'{data_dir}/wind_data_{dataset}.pickle' 

    # pandas dataframe
    data_puffs = pd.read_pickle(puff_filename)
    data_wind = pd.read_pickle(wind_filename)

    # Load plume/wind data and truncate away upto t_val_min 
    if t_val_min is not None:
        data_wind.query("time >= {}".format(t_val_min), inplace=True)
        data_puffs.query("time >= {}".format(t_val_min), inplace=True)

    # SPEEDUP: **Further** truncate plume/wind data by sim. time
    if t_val_max is not None:
        data_wind.query("time <= {}".format(t_val_max), inplace=True)
        data_puffs.query("time <= {}".format(t_val_max), inplace=True)

    ## Downsample to env_dt!
    env_dt_int = int(env_dt*100)
    assert env_dt_int in [2, 4, 5, 10] # Limit downsampling to these for now!
    if 'tidx' not in data_wind.columns:
        data_wind['tidx'] = (data_wind['time']*100).astype(int)
    if 'tidx' not in data_puffs.columns:
        data_puffs['tidx'] = (data_puffs['time']*100).astype(int)
    data_wind.query("tidx % @env_dt_int == 0", inplace=True)
    data_puffs.query("tidx % @env_dt_int == 0", inplace=True)

    # Sparsify puff data (No change in wind)
    if puff_sparsity < 0.99:
        print(f"[load_plume] Sparsifying puffs to {puff_sparsity}x")
        puff_sparsity = np.clip(puff_sparsity, 0.0, 1.0)
        drop_idxs = data_puffs['puff_number'].unique()
        drop_idxs = pd.Series(drop_idxs).sample(frac=(1.00-puff_sparsity))
        data_puffs.query("puff_number not in @drop_idxs", inplace=True)

    # Multiply radius 
    if radius_multiplier != 1.0:
        print("Applying radius_multiplier", radius_multiplier)
        data_puffs.loc[:,'radius'] *= radius_multiplier

    min_radius = 0.01

    # Adjust diffusion rate
    if diffusion_multiplier != 1.0:
        print("Applying diffusion_multiplier", diffusion_multiplier)
        data_puffs.loc[:,'radius'] -= min_radius # subtract initial radius
        data_puffs.loc[:,'radius'] *= diffusion_multiplier # adjust 
        data_puffs.loc[:,'radius'] += min_radius # add back initial radius

    # Add other columns
    data_puffs['x_minus_radius'] = data_puffs.x - data_puffs.radius
    data_puffs['x_plus_radius'] = data_puffs.x + data_puffs.radius
    data_puffs['y_minus_radius'] = data_puffs.y - data_puffs.radius
    data_puffs['y_plus_radius'] = data_puffs.y + data_puffs.radius
    data_puffs['concentration'] = (min_radius/data_puffs.radius)**3


    return data_puffs, data_wind


def rotate_wind_optimized(data_wind, rotation_angle_degrees, mirror):
    """
    Rotate wind direction vectors by specified angle around origin.
    Optimized for angles: [0, 90, 180, -90] degrees.
    
    Parameters:
    -----------
    data_wind : pd.DataFrame
        Wind dataframe with columns: wind_x, wind_y, time, tidx
    rotation_angle_degrees : float
        Rotation angle in degrees (0, 90, 180, or -90)
    
    Returns:
    --------
    pd.DataFrame: Rotated wind dataframe
    """
    
    if rotation_angle_degrees is None:
        # No rotation needed
        return data_wind
    
    # Copy dataframe to avoid modifying original
    wind_rotated = data_wind.copy()
    
    if rotation_angle_degrees == 0:
        # No rotation needed
        if mirror:
            # Mirror along the long side 
            wind_rotated['wind_y'] = -wind_rotated['wind_y']
        else:
            # No mirroring or rotating, just return original
            return wind_rotated
    elif rotation_angle_degrees == 90:
        # 90° rotation: (x,y) -> (-y,x)
        wind_x_new = -wind_rotated['wind_y']
        wind_y_new = wind_rotated['wind_x']
        if mirror:
            # Mirror along the long side
            wind_x_new = -wind_x_new
    elif rotation_angle_degrees == 180:
        # 180° rotation: (x,y) -> (-x,-y)
        wind_x_new = -wind_rotated['wind_x']
        wind_y_new = -wind_rotated['wind_y']
        if mirror:
            # Mirror along the long side
            wind_y_new = -wind_y_new
    elif rotation_angle_degrees == -90:
        # -90° rotation: (x,y) -> (y,-x)
        wind_x_new = wind_rotated['wind_y']
        wind_y_new = -wind_rotated['wind_x']
        if mirror:
            # Mirror along the long side
            wind_x_new = -wind_x_new
    else:
        raise ValueError(f"Unsupported rotation angle: {rotation_angle_degrees}. "
                        "Supported angles are: [0, 90, 180, -90]")
    
    wind_rotated['wind_x'] = wind_x_new
    wind_rotated['wind_y'] = wind_y_new
    
    return wind_rotated


def rotate_puffs_optimized(data_puffs, rotation_angle_degrees, mirror):
    """
    Rotate puff locations by specified angle around origin.
    Optimized for angles: [0, 90, 180, -90] degrees, or None.
    
    Parameters:
    -----------
    data_puffs : pd.DataFrame  
        Puff dataframe with columns: puff_number, time, x, y, radius, tidx,
        x_minus_radius, x_plus_radius, y_minus_radius, y_plus_radius, concentration
    rotation_angle_degrees : float
        Rotation angle in degrees (0, 90, 180, or -90)
    
    Returns:
    --------
    pd.DataFrame: Rotated puffs dataframe
    """
    
    # Copy dataframe to avoid modifying original
    if rotation_angle_degrees is None:
        # No rotation needed
        return data_puffs
    puffs_rotated = data_puffs.copy()
    if rotation_angle_degrees == 0:
        # No rotation needed
        if mirror:
            # Mirror along the long side 
            puffs_rotated['y'] = -puffs_rotated['y']
        else:
            # No mirroring or rotating, just return original
            return puffs_rotated
    elif rotation_angle_degrees == 90:
        # 90° rotation: (x,y) -> (-y,x)
        x_new = -puffs_rotated['y']
        y_new = puffs_rotated['x']
        if mirror:
            # Mirror along the long side
            x_new = -x_new
    elif rotation_angle_degrees == 180:
        # 180° rotation: (x,y) -> (-x,-y)
        x_new = -puffs_rotated['x']
        y_new = -puffs_rotated['y']
        if mirror:
            # Mirror along the long side
            y_new = -y_new
    elif rotation_angle_degrees == -90:
        # -90° rotation: (x,y) -> (y,-x)
        x_new = puffs_rotated['y']
        y_new = -puffs_rotated['x']
        if mirror:
            # Mirror along the long side
            x_new = -x_new
    else:
        raise ValueError(f"Unsupported rotation angle: {rotation_angle_degrees}. "
                        "Supported angles are: [0, 90, 180, -90]")
    
    puffs_rotated['x'] = x_new
    puffs_rotated['y'] = y_new
    
    # Update radius-based columns
    puffs_rotated['x_minus_radius'] = puffs_rotated['x'] - puffs_rotated['radius']
    puffs_rotated['x_plus_radius'] = puffs_rotated['x'] + puffs_rotated['radius']
    puffs_rotated['y_minus_radius'] = puffs_rotated['y'] - puffs_rotated['radius']
    puffs_rotated['y_plus_radius'] = puffs_rotated['y'] + puffs_rotated['radius']
    
    return puffs_rotated


def rotate_wind(data_wind, rotation_angle_degrees):
    """
    Rotate wind direction vectors by specified angle around origin.
    
    Parameters:
    -----------
    data_wind : pd.DataFrame
        Wind dataframe with columns: wind_x, wind_y, time, tidx
    rotation_angle_degrees : float
        Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
    --------
    pd.DataFrame: Rotated wind dataframe
    """
    
    # Convert angle to radians
    theta = np.radians(rotation_angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Copy dataframe to avoid modifying original
    wind_rotated = data_wind.copy()
    
    # Rotate wind direction vectors
    wind_x_new = wind_rotated['wind_x'] * cos_theta - wind_rotated['wind_y'] * sin_theta
    wind_y_new = wind_rotated['wind_x'] * sin_theta + wind_rotated['wind_y'] * cos_theta
    
    wind_rotated['wind_x'] = wind_x_new
    wind_rotated['wind_y'] = wind_y_new
    
    return wind_rotated

def rotate_puffs(data_puffs, rotation_angle_degrees):
    """
    Rotate puff locations by specified angle around origin.
    
    Parameters:
    -----------
    data_puffs : pd.DataFrame  
        Puff dataframe with columns: puff_number, time, x, y, radius, tidx,
        x_minus_radius, x_plus_radius, y_minus_radius, y_plus_radius, concentration
    rotation_angle_degrees : float
        Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
    --------
    pd.DataFrame: Rotated puffs dataframe
    """
    
    # Convert angle to radians
    theta = np.radians(rotation_angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Copy dataframe to avoid modifying original
    puffs_rotated = data_puffs.copy()
    
    # Rotate puff positions
    x_new = puffs_rotated['x'] * cos_theta - puffs_rotated['y'] * sin_theta
    y_new = puffs_rotated['x'] * sin_theta + puffs_rotated['y'] * cos_theta
    
    puffs_rotated['x'] = x_new
    puffs_rotated['y'] = y_new
    
    # Update radius-based columns
    puffs_rotated['x_minus_radius'] = puffs_rotated['x'] - puffs_rotated['radius']
    puffs_rotated['x_plus_radius'] = puffs_rotated['x'] + puffs_rotated['radius']
    puffs_rotated['y_minus_radius'] = puffs_rotated['y'] - puffs_rotated['radius']
    puffs_rotated['y_plus_radius'] = puffs_rotated['y'] + puffs_rotated['radius']
    
    return puffs_rotated

def rotate_wind_and_puffs(data_wind, data_puffs, rotation_angle_degrees):
    """
    Rotate both wind direction and puff locations by the same angle around origin.
    Convenience function that calls both rotate_wind and rotate_puffs.
    
    Parameters:
    -----------
    data_wind : pd.DataFrame
        Wind dataframe with columns: wind_x, wind_y, time, tidx
    data_puffs : pd.DataFrame  
        Puff dataframe with columns: puff_number, time, x, y, radius, tidx,
        x_minus_radius, x_plus_radius, y_minus_radius, y_plus_radius, concentration
    rotation_angle_degrees : float
        Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
    --------
    tuple: (rotated_wind_df, rotated_puffs_df)
    """
    
    wind_rotated = rotate_wind(data_wind, rotation_angle_degrees)
    puffs_rotated = rotate_puffs(data_puffs, rotation_angle_degrees)
    
    return wind_rotated, puffs_rotated

def get_concentration_at_tidx(data, tidx, x_val, y_val, rotate_by=0, mirror=False):
    # find the indices for all puffs that intersect the given x,y,time point
    qx = str(x_val) + ' > x_minus_radius and ' + str(x_val) + ' < x_plus_radius'
    qy = str(y_val) + ' > y_minus_radius and ' + str(y_val) + ' < y_plus_radius'
    q = qx + ' and ' + qy
    if rotate_by:
        data_rot = rotate_puffs_optimized(data[data.tidx==tidx], rotate_by, mirror)
        d = data_rot.query(q)
        # t_val = wind[wind.tidx==tidx].time.values[0] 
        # print("d.concentration.sum()", d.concentration.sum())
        # if d.concentration.sum() < config.env['odor_threshold']:
        #     print("No puffs at this location and time", tidx, x_val, y_val)
        #     fig, ax = sim_analysis.plot_puffs_and_wind_vectors(
        #         data_rot, 
        #         wind, 
        #         t_val, 
        #         fname='/src/tamagotchi/puffs_and_wind_vectors_initial.png', 
        #         plotsize=(8,8))
        #     # plot all start locations
        #     ax.scatter(x_val, y_val, c='red', s=2, label='Start Locations')
        #     ax.legend()
        #     fig.savefig('/src/tamagotchi/puffs_and_wind_vectors_initial.png')
    else:
        d = data[data.tidx==tidx].query(q)
    return d.concentration.sum()

def cleanup_log_dir(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    return
    # try:
    #     os.makedirs(log_dir)
    # except OSError:
    #     files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    #     for f in files:
    #         os.remove(f)
    
def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("Learning rate: ", lr, flush=True)
    
def update_cosine_restart_schedule(optimizer, epoch, initial_lr, restart_period=100):
    t = epoch % restart_period
    lr = initial_lr * 0.5 * (1 + math.cos(math.pi * t / restart_period))
    for group in optimizer.param_groups:
        group['lr'] = lr
    print("Epoch {} Learning rate: {}".format(epoch, lr), flush=True)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# New version: One circle at hardcoded (x,y) with quiver 
def plot_wind_vectors(data_puffs, data_wind, t_val, ax, invert_colors=False):
    # Get mean wind vector at given time
    data_at_t = data_wind[data_wind.time == t_val]
    v_x, v_y = data_at_t.wind_x.mean(), data_at_t.wind_y.mean()
    
    # Normalize wind vector (direction only)
    norm = np.sqrt(v_x ** 2 + v_y ** 2)
    if norm < 1e-8:
        v_x, v_y = 0.0, 0.0  # avoid division by zero
    else:
        v_x, v_y = v_x / norm, v_y / norm
    
    # Arrow location
    x, y = -0.15, 0.6
    color = 'white' if invert_colors else 'black'

    # Draw wind vector (quiver automatically handles direction signs)
    ax.quiver(x, y, v_x, v_y, color=color, scale=5, scale_units='xy', angles='xy', width=0.01)

    # Draw wind circle
    ax.scatter(x, y, s=500,
               facecolors='none',
               edgecolors=color,
               linestyle='--')
    
    return ax

def plot_puffs(data, t_val, ax=None, fig=None, show=True):
    # TODO check color to concentration mapping
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    elif fig is None:
        fig = ax.figure
        
    # xmin = -2 #data.x.min()
    # xmax = 12 #data.x.max()
    # ymin = -5 #data.y.min()
    # ymax = +5 #data.y.max()
    # set limits
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # ax.set_aspect('equal') # move into plot_puffs_and_wind_vectors - keep here for record keeping

    # data_at_t = data[data.time==t_val] # Float equals is dangerous!
    data_at_t = data[np.isclose(data.time, t_val, atol=1e-3)] # Smallest dt=0.01, so this is more than enough!
    # print("data_at_t.shape", data_at_t.shape, t_val, data.time.min(), data.time.max())

    c = data_at_t.concentration
    # print(c, t_val)

    # alphas = (np.log(c+1e-5)+np.abs(np.log(1e-5))).values
    # alphas /= np.max(alphas)
    # alphas = np.clip(alphas, 0.0, 1.0)

    alphas = c.values
    alphas /= np.max(alphas) # 0...1
    alphas = np.power(alphas, 1/8) # See minimal2 notebook
    # alphas = np.power(alphas, 10)
    alphas = np.clip(alphas, 0.2, 0.4)
    # decay alpha by distance too
    distance_from_source = np.sqrt(data_at_t.x**2 + data_at_t.y**2)
    alphas *= 2.5/distance_from_source
    alphas = np.clip(alphas, 0.05, 0.4)


    rgba_colors = np.zeros((data_at_t.time.shape[0],4))
    # rgba_colors[:,0] = 1.0 # Red
    # rgba_colors[:,2] = 1.0 # Blue
    # https://matplotlib.org/3.1.1/gallery/color/named_colors.html
    # https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    rgba_colors[:,0:3] = matplotlib.colors.to_rgba('gray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkgray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('dimgray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkslategray')[:3] # too dark
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightsteelblue')[:3] # ok
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('red')[:3] 
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightskyblue')[:3] 

    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas

    # fig.canvas.draw()
    # s = ((ax.get_window_extent().width  / (xmax-xmin+1.) * 72./fig.dpi) ** 2)
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_width = bbox.width  # in inches

    k = 6250 * ((ax_width / 8.0) ** 2)  # calibrate denominator as needed
    s = k * (data_at_t.radius)**2
    # print('size', s) # 885
    ax.scatter(data_at_t.x, data_at_t.y, s=s, facecolor=rgba_colors, edgecolor='none')
    
    # 070225 - branch for using radius as size 
    # from matplotlib.patches import Circle
    # for _, row in data_at_t.iterrows():
    #     circle = Circle(
    #         (row['x'], row['y']),
    #         radius=row['radius'],  # This is in data units!
    #         facecolor=plt.cm.viridis(row['concentration'] / data_at_t['concentration'].max()),
    #         edgecolor='none',
    #         alpha=0.7
    #     )
    #     ax.add_patch(circle)

    # ax.set_xlim(data_at_t['x'].min() - 1, data_at_t['x'].max() + 1)
    # ax.set_ylim(data_at_t['y'].min() - 1, data_at_t['y'].max() + 1)
    # ax.set_aspect('equal')
    # plt.tight_layout()
    
    if show:
        plt.show()
    return ax

def plot_puffs_and_wind_vectors(data_puffs, data_wind, t_val, ax=None, fig=None, fname='', plotsize=(10,10), aspect_ratio=False, show=True, invert_colors=False):
    if fig is None:
        fig = plt.figure(figsize=plotsize)
    if ax is None:
        ax = fig.add_subplot(111)
    ax = plot_wind_vectors(data_puffs, data_wind, t_val, ax, invert_colors=invert_colors)
    ax = plot_puffs(data_puffs, t_val, ax=ax, fig=fig, show=False)
    ax.patch.set_facecolor('none') 
    if aspect_ratio:
        ax.set_aspect(aspect_ratio)
    else:
        ax.set_aspect('equal')
    if len(fname) > 0:
        # fname = savedir + '/' + 'puff_animation_' + str(idx).zfill(int(np.log10(data['puffs'].shape[1]))+1) + '.jpg'
        fig.savefig(fname, format='jpg', bbox_inches='tight')
        plt.close()
    return fig, ax

def save_tc_schedule(schedule, num_updates, num_processes, num_steps, save_dir):
    df_schedule = pd.DataFrame(schedule)
    df_schedule.sort_index(axis=0, inplace=True)
    df_schedule.loc[num_updates] = None
    df_schedule.fillna(method='ffill', inplace=True)
    df_schedule['update'] = df_schedule.index
    df_schedule['timestep'] = df_schedule['update'] * num_processes * num_steps
    matplotlib.use('Agg') # no not display plots
    ax = df_schedule.plot(x="timestep", y="birthx", legend=False)
    ax2 = ax.twinx()
    df_schedule.plot(x="timestep", y="wind_cond", ax=ax2, legend=False, color="r")
    ax.scatter(df_schedule["timestep"], df_schedule["birthx"])
    ax2.scatter(df_schedule["timestep"], df_schedule["wind_cond"], color="r")
    ax2.set_yticks([1, 2, 3])
    ax.figure.legend()
    ax.set_yscale('log')
    
    df_schedule.to_csv(os.path.join(save_dir, 'json', 'schedule.tsv'), sep='\t', index=False)
    plt.savefig(os.path.join(save_dir, 'json', 'schedule.png'))
    
def plot_tc_schedule(schedule, num_updates, num_processes, num_steps):
    """
    Plot the curriculum learning schedule. Returns a figure object without displaying it.
    """
    df_schedule = pd.DataFrame(schedule)
    df_schedule.sort_index(axis=0, inplace=True)
    df_schedule.loc[num_updates] = None
    df_schedule.fillna(method='ffill', inplace=True)
    df_schedule['update'] = df_schedule.index
    df_schedule['timestep'] = df_schedule['update'] * num_processes * num_steps
    matplotlib.use('Agg') # no not display plots
    ax = df_schedule.plot(x="timestep", y="birthx", legend=False)
    ax2 = ax.twinx()
    df_schedule.plot(x="timestep", y="wind_cond", ax=ax2, legend=False, color="r")
    ax.scatter(df_schedule["timestep"], df_schedule["birthx"])
    ax2.scatter(df_schedule["timestep"], df_schedule["wind_cond"], color="r")
    ax2.set_yticks([1, 2, 3])
    ax.figure.legend()
    ax.set_yscale('log')
    fig = ax.get_figure()
    return fig


# for logging episode statistics 
def update_eps_info(update_episodes_df, info, episode_counter, update_idx):
    # update the episode statistics
    update_episodes_df = pd.concat([update_episodes_df,pd.DataFrame([
        {
            'episode_id': episode_counter,
            'dataset': info['dataset'],
            'outcome': info['done'],
            'reward': info['episode']['r'],
            'plume_density': info['plume_density'],
            'start_tidx': info['step_offset'],
            'end_tidx': info['tidx'],
            'location_initial': info['location_initial'],
            'stray_initial': info['stray_initial'],
            'end_location': info['location'],
            'init_angle': info['init_angle'],
            'rotate_by': info['rotate_by'],
            'mirror': info['mirror'],
            'update_idx': update_idx,
        }])])
    return update_episodes_df


def log_agent_learning(j, advantages, value_loss, action_loss, dist_entropy, clip_fraction, learning_rate, use_mlflow=True):
    if not use_mlflow:
        return
    mlflow.log_metric("advantages_mean", advantages.mean().item(), step=j)
    mlflow.log_metric("advantages_std", advantages.std().item(), step=j)
    mlflow.log_metric("advantages_max", advantages.max().item(), step=j)
    mlflow.log_metric("advantages_min", advantages.min().item(), step=j)
    mlflow.log_metric("value_loss", value_loss, step=j)
    mlflow.log_metric("action_loss", action_loss, step=j)
    mlflow.log_metric("dist_entropy", dist_entropy, step=j)
    mlflow.log_metric("clip_fraction", clip_fraction, step=j)
    mlflow.log_metric("learning_rate", learning_rate, step=j)


def log_agent_learning_wind_obsver(j, advantages, value_loss, action_loss, dist_entropy, clip_fraction, learning_rate, aux_loss_dict, use_mlflow=True):
    if not use_mlflow:
        return
    mlflow.log_metric("ppo/advantages_mean", advantages.mean().item(), step=j)
    mlflow.log_metric("ppo/advantages_std", advantages.std().item(), step=j)
    mlflow.log_metric("ppo/advantages_max", advantages.max().item(), step=j)
    mlflow.log_metric("ppo/advantages_min", advantages.min().item(), step=j)
    mlflow.log_metric("ppo/value_loss", value_loss, step=j)
    mlflow.log_metric("ppo/action_loss", action_loss, step=j)
    mlflow.log_metric("ppo/dist_entropy", dist_entropy, step=j)
    mlflow.log_metric("ppo/clip_fraction", clip_fraction, step=j)
    mlflow.log_metric("ppo/learning_rate", learning_rate, step=j)

    all_wind_nll = aux_loss_dict['wind_nll_all']
    all_wind_sqerr = aux_loss_dict['wind_sqerr_all']
    all_wind_logvar = aux_loss_dict['wind_logvar_all']
    wind_nll_mean = all_wind_nll.mean().item()
    wind_nll_std  = all_wind_nll.std().item()
    wind_loss_epoch = aux_loss_dict["wind_loss_epoch"]
    mlflow.log_metric("ppo/wind_loss_mean", wind_loss_epoch, step=j)
    mlflow.log_metric("ppo/wind_nll_mean", wind_nll_mean, step=j)
    mlflow.log_metric("ppo/wind_nll_std",  wind_nll_std, step=j)
    mlflow.log_metric("ppo/wind_sqerr_mean", all_wind_sqerr.mean().item(), step=j)
    mlflow.log_metric("ppo/wind_logvar_mean", all_wind_logvar.mean().item(), step=j)

    

def log_eps_artifacts(j, args, update_episodes_df, use_mlflow=True):
    """
    Log episode statistics and plot a histogram of plume density for successful episodes.
    Args:
        j (int): Update index for logging and for labeling the plot.
        args (Namespace): Contains `save_dir` for saving the plot.
        update_episodes_df (pd.DataFrame): DataFrame with 'outcome', 'dataset', and 'plume_density'.
    """
    
    # Log episode statistics
    if use_mlflow:
        for outcome in ['HOME', 'OOB', 'OOT']:
            mlflow.log_metric(f"{outcome}_num", sum(update_episodes_df['outcome'] == outcome), step=j)
            mlflow.log_metric(f"{outcome}_ratio", sum(update_episodes_df['outcome'] == outcome) / len(update_episodes_df['outcome']), step=j)
            mlflow.log_metric('num_episodes', len(update_episodes_df['outcome']), step=j)
    log_path = f"{args.save_dir}/tmp/{args.model_fname}_eps_log_{j}.csv"
    update_episodes_df.to_csv(log_path, index=False)
    if use_mlflow:
        try:
            mlflow.log_artifact(log_path, artifact_path=f"eps_log")
        except Exception as e:
            print(f"Error logging artifact {log_path}: {e}")
        os.remove(log_path)
    
    # Plot plume density histogram for successful episodes
    successful_df = update_episodes_df[update_episodes_df['outcome'] == 'HOME']
    # Check if there's any data to plot
    if len(successful_df) > 0:        
        # Plot success rate by plume density and dataset
        # Define common bins for plume density
        bins = np.linspace(update_episodes_df['plume_density'].min(), 
                        update_episodes_df['plume_density'].max(), 10)
        bin_width = bins[1] - bins[0]
        # Get unique datasets
        datasets = update_episodes_df['dataset'].unique()
        # Set up the plot
        plt.figure(figsize=(4, 4))
        # Offset width to prevent bar overlap
        offset_factor = 0.8 / len(datasets)

        for i, dataset in enumerate(datasets):
            subset = update_episodes_df[update_episodes_df['dataset'] == dataset].copy()
            subset['plume_bin'] = pd.cut(subset['plume_density'], bins=bins)
            
            grouped = subset.groupby('plume_bin')
            
            # Compute success rate and number of successes
            success_rate = grouped['outcome'].apply(lambda x: (x == 'HOME').mean())
            n_success = grouped['outcome'].apply(lambda x: (x == 'HOME').sum())
            bin_centers = grouped['plume_density'].apply(lambda x: x.mean()).values
            
            # Apply offset to bin centers to avoid bar overlap
            bin_centers_shifted = bin_centers + (i - len(datasets)/2) * offset_factor * bin_width

            # Plot bars
            plt.bar(bin_centers_shifted, success_rate.values, 
                    width=offset_factor * bin_width, 
                    label=dataset, alpha=0.8, color=config.mlflow_colors[dataset])

            # Annotate number of successes, colored by group
            for x, y, n in zip(bin_centers_shifted, success_rate.values, n_success.values):
                if np.isfinite(x) and np.isfinite(y):
                    plt.text(x, y + 0.02, str(n), ha='center', va='bottom', fontsize=8, color=config.mlflow_colors[dataset])

        # Labels and formatting
        plt.xlabel('Plume Density')
        plt.ylabel('HOME fraction; n = {}'.format(len(update_episodes_df['outcome'])))
        plt.title(f"Success Rate by Plume Density and Dataset (Update {int(update_episodes_df['update_idx'].min())} - {int(update_episodes_df['update_idx'].max())} )")
        plt.ylim(0, 1.05)
        plt.legend(title='Dataset')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Save the figure to a file in your update directory
        plt_path = f"{args.save_dir}/tmp/{args.model_fname.replace('.pt', '_')}HOME_density_{j}_rate.png"
        plt.savefig(plt_path, dpi=100, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        if use_mlflow: 
            try:
                mlflow.log_artifact(plt_path, artifact_path=f"figs")
            except Exception as e:
                print(f"Error logging artifact {plt_path}: {e}")
            os.remove(plt_path)
        

# from a2c_ppo_acktr/storage.py
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])
