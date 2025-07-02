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
    Optimized for angles: [0, 90, 180, -90] degrees.
    
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

    alphas *= 2.5/data_at_t.x # decay alpha by distance too
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
def update_eps_info(update_episodes_df, infos, episode_counter):
    # update the episode statistics
    for i in range(len(infos)):
        if infos[i]['done']:
            update_episodes_df = pd.concat([update_episodes_df,pd.DataFrame([
                {
                    'episode_id': episode_counter,
                    'dataset': infos[i]['dataset'],
                    'outcome': infos[i]['done'],
                    'reward': infos[i]['episode']['r'],
                    'plume_density': infos[i]['plume_density'],
                    'start_tidx': infos[i]['step_offset'],
                    'end_tidx': infos[i]['tidx'],
                    'location_initial': infos[i]['location_initial'],
                    'end_location': infos[i]['location'],
                    'init_angle': infos[i]['init_angle'],
                    'rotate_by': infos[i]['rotate_by'],
                    'mirror': infos[i]['mirror'],
                }])]
            )
    return update_episodes_df


def log_agent_learning(j, value_loss, action_loss, dist_entropy, clip_fraction, learning_rate, use_mlflow=True):
    if not use_mlflow:
        return
    mlflow.log_metric("value_loss", value_loss, step=j)
    mlflow.log_metric("action_loss", action_loss, step=j)
    mlflow.log_metric("dist_entropy", dist_entropy, step=j)
    mlflow.log_metric("clip_fraction", clip_fraction, step=j)
    mlflow.log_metric("learning_rate", learning_rate, step=j)

            

def log_eps_artifacts(j, args, update_episodes_df, use_mlflow=True):
    """
    Log episode statistics and plot a histogram of plume density for successful episodes.
    Args:
        j (int): Update index for logging and for labeling the plot.
        args (Namespace): Contains `save_dir` for saving the plot.
        update_episodes_df (pd.DataFrame): DataFrame with 'outcome', 'dataset', and 'plume_density'.
    """
    
    # Log episode statistics
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
        plt.ylabel('Success Rate (Fraction of HOME outcomes)')
        plt.title(f'Success Rate by Plume Density and Dataset (Update {j})')
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


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[
                        step + 1] * self.masks[step +
                                               1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
