from tamagotchi import config
from tamagotchi import sim_utils as sim_analysis
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from tamagotchi import wb as mlflow  # wandb shim with mlflow API
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
    try:
        data_wind = pd.read_pickle(wind_filename)
    except Exception as e:
        # Try to load Toha data format where all data is in one file
        # pandas dataframe - Toha data has plume and wind in one file
        data = pd.read_pickle(puff_filename)
        data_puffs = data.copy()
        # take only puff columns
        puff_columns = ['puff_number', 'time', 'puff_x', 'puff_y', 'puff_r']
        data_puffs = data_puffs[puff_columns]
        # rename columns
        data_puffs.rename(columns={'puff_x':'x', 'puff_y':'y', 'puff_r':'radius'}, inplace=True)
        # make the wind df
        data_wind = data.copy()
        wind_columns = ['wind_x', 'wind_y', 'time']
        data_wind = data_wind[wind_columns]

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
    epsilon = 1e-4
    data_puffs['concentration'] = (min_radius/(data_puffs.radius + epsilon))**3


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
        theta = np.deg2rad(rotation_angle_degrees)
        c, s = np.cos(theta), np.sin(theta)
        wind_x_new = c * wind_rotated['wind_x'] - s * wind_rotated['wind_y']
        wind_y_new = s * wind_rotated['wind_x'] + c * wind_rotated['wind_y']
        if mirror:
            wind_x_new = -wind_x_new
    
    wind_rotated['wind_x'] = wind_x_new
    wind_rotated['wind_y'] = wind_y_new
    
    return wind_rotated


def rotate_puffs_optimized(data_puffs, rotation_angle_degrees, mirror):
    """
    Rotate puff locations by specified angle around origin.
    Optimized for angles: [0, 90, 180, -90] degrees, or None.
    Used when making the traj vis and another copy of this in the env obj
    
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
        theta = np.deg2rad(rotation_angle_degrees)
        c, s = np.cos(theta), np.sin(theta)
        x_new = c * puffs_rotated['x'] - s * puffs_rotated['y']
        y_new = s * puffs_rotated['x'] + c * puffs_rotated['y']
        if mirror:
            x_new = -x_new

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
        try:
            d = data[data.tidx==tidx].query(q)
        except Exception as e:
            raise ValueError(f"Error occurred while querying data for tidx {tidx}: {e}; query: {q}")
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

# Wind direction indicator: a dashed circle + arrow drawn in axes-fraction
# coordinates, so it stays a fixed visible size regardless of the arena/axis
# limits (which can vary widely, e.g. when sized per-episode from x0/y0).
def plot_wind_vectors(data_puffs, data_wind, t_val, ax, invert_colors=False, wind_vector=True):
    # Get mean wind vector at given time (tolerant match: t_val may not equal a
    # stored time exactly due to float precision / downsampling).
    data_at_t = data_wind[np.isclose(data_wind.time, t_val, atol=1e-3)]
    v_x, v_y = data_at_t.wind_x.mean(), data_at_t.wind_y.mean()

    # Normalize wind vector (direction only)
    norm = np.sqrt(v_x ** 2 + v_y ** 2)
    if not np.isfinite(norm) or norm < 1e-8:
        v_x, v_y = 0.0, 0.0  # avoid division by zero / NaN (no rows at t_val)
    else:
        v_x, v_y = v_x / norm, v_y / norm

    # Indicator placement/size in axes fraction (independent of data limits)
    cx, cy = 0.12, 0.8   # circle center (top-left corner)
    L = 0.032             # arrow half-length
    color = 'white' if invert_colors else 'black'

    if not wind_vector:
        return ax

    # Draw wind vector as an arrow through the circle center
    ax.annotate('', xy=(cx + v_x * L, cy + v_y * L), xytext=(cx, cy),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2), zorder=6)

    # Draw wind circle
    ax.scatter([cx], [cx], s=900,
               facecolors='none',
               edgecolors=color,
               linestyle='--',
               transform=ax.transAxes,
               zorder=5)

    return ax

def plot_puffs(data, t_val, ax=None, fig=None, show=True, scatter_size_factor=None):
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
    if scatter_size_factor is not None:
        k = scatter_size_factor
    else:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        k = 6250 * ((bbox.width / 8.0) ** 2)
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

def plot_agent_traj(traj_df, t_val, ax, ep_idx=None, seed=None, **kwargs):
    """Plot agent trajectory up to t_val, coloring dots by on/off plume state.

    Args:
        traj_df: DataFrame with columns t_val, loc_x, loc_y, odor_eps_log (or similar).
                 Filter to a single episode before calling, or pass ep_idx/seed to filter here.
        t_val:   Current animation time; only steps up to this time are drawn.
        ax:      Matplotlib axes to draw on.
        ep_idx:  If provided, filter traj_df to rows where ep_idx == ep_idx.
        seed:    If provided, additionally filter to rows where seed == seed.
    """
    df = traj_df.copy()
    if ep_idx is not None:
        df = df[df['ep_idx'] == ep_idx]
    if seed is not None:
        df = df[df['seed'] == seed]

    df = df[df['t_val'] <= t_val]
    if df.empty:
        return ax

    # Trail line
    linecolor = config.traj_colormap['off']
    ax.plot(df['loc_x'], df['loc_y'], c=linecolor, lw=0.5, zorder=2)

    # Per-step scatter colored by odor state
    if 'odor_eps_log' in df.columns:
        colors = [
            config.traj_colormap['off'] if x <= config.env['odor_threshold']
            else config.traj_colormap['on']
            for x in df['odor_eps_log']
        ]
    else:
        colors = linecolor

    fontsize = kwargs.get('fontsize', None)
    scatter_kw = dict(s=4, zorder=3, linewidths=0)
    ax.scatter(df['loc_x'], df['loc_y'], c=colors, **scatter_kw)

    # Mark start
    ax.scatter(df['loc_x'].iloc[0], df['loc_y'].iloc[0],
               c='black', s=20, zorder=4, marker='o')

    ax.set_xlabel('Arena length [m]')
    ax.set_ylabel('Arena width [m]')
    return ax


def make_animation_update(time_values, data_puffs, data_wind, traj_df,
                          ax, ep_idx=None, seed=None,
                          xlim=(-2, 12), ylim=(-5, 5)):
    """Return an update(i) closure suitable for FuncAnimation.

    Precomputes per-frame lookups (closest wind time, puff slices, trajectory
    cumulative colors) so the hot loop does minimal work.

    Args:
        time_values: 1-D array of timestamps, one per animation frame.
        data_puffs:  Puff dataframe (pre-filtered to the episode time range).
        data_wind:   Wind dataframe.
        traj_df:     Agent trajectory dataframe (filtered or full).
        ax:          Axes to draw on (cleared each frame).
        ep_idx:      Episode index passed through to plot_agent_traj.
        seed:        Seed passed through to plot_agent_traj.
        xlim/ylim:   Axis limits as (min, max) tuples.
    """
    fig = ax.figure

    # --- precompute closest wind time for every frame (vectorized) -----------
    wind_times = np.sort(data_wind['time'].unique())
    idx = np.searchsorted(wind_times, time_values, side='left')
    idx = np.clip(idx, 1, len(wind_times) - 1)
    left_t = wind_times[idx - 1]
    right_t = wind_times[idx]
    closest_t_arr = np.where(
        np.abs(time_values - left_t) <= np.abs(time_values - right_t),
        left_t, right_t
    )

    # --- precompute puff slices grouped by time (O(1) dict lookup per frame) -
    puffs_by_time = {t: grp for t, grp in data_puffs.groupby('time')}

    # --- precompute scatter size factor once (avoids canvas call per frame) --
    fig.canvas.draw()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    k = 6250 * ((bbox.width / 8.0) ** 2)

    # --- precompute trajectory for this episode, sorted by t_val -------------
    ep_traj = traj_df.copy()
    if ep_idx is not None:
        ep_traj = ep_traj[ep_traj['ep_idx'] == ep_idx]
    if seed is not None:
        ep_traj = ep_traj[ep_traj['seed'] == seed]
    ep_traj = ep_traj.sort_values('t_val').reset_index(drop=True)
    traj_times = ep_traj['t_val'].values

    odor_threshold = config.env['odor_threshold']
    color_on = config.traj_colormap['on']
    color_off = config.traj_colormap['off']
    if 'odor_eps_log' in ep_traj.columns:
        traj_colors = np.where(
            ep_traj['odor_eps_log'].values > odor_threshold,
            color_on, color_off
        )
    else:
        traj_colors = np.full(len(ep_traj), color_off)

    def update(i):
        ax.clear()
        closest_t = closest_t_arr[i]

        # wind vector
        plot_wind_vectors(data_puffs, data_wind, closest_t, ax)

        # puffs — pass pre-sliced group and cached scatter size
        puff_slice = puffs_by_time.get(closest_t)
        if puff_slice is not None:
            plot_puffs(puff_slice, closest_t, ax, show=False,
                       scatter_size_factor=k)

        # trajectory up to current time via searchsorted
        end = int(np.searchsorted(traj_times, closest_t, side='right'))
        if end > 0:
            df_slice = ep_traj.iloc[:end]
            c_slice = traj_colors[:end]
            ax.plot(df_slice['loc_x'], df_slice['loc_y'],
                    c=color_off, lw=0.5, zorder=2)
            ax.scatter(df_slice['loc_x'], df_slice['loc_y'],
                       c=c_slice, s=4, zorder=3, linewidths=0)
            ax.scatter(df_slice['loc_x'].iloc[0], df_slice['loc_y'].iloc[0],
                       c='black', s=20, zorder=4, marker='o')

        ax.set_title(f'ep={ep_idx} seed={seed} t={closest_t:.2f}s')
        ax.set_xlabel('Arena length [m]')
        ax.set_ylabel('Arena width [m]')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.plot([0, 0],[-0.3,+0.3],'k-', linestyle = ":", lw=2) # manuscript white background, black lines
        ax.plot([-0.3,+0.3],[0, 0],'k-', linestyle = ":", lw=2)


    return update



def plot_puffs_and_wind_vectors(data_puffs, data_wind, t_val, ax=None, fig=None, fname='', plotsize=(10,10), aspect_ratio=False, show=True, invert_colors=False, wind_vector=True):
    if fig is None:
        fig = plt.figure(figsize=plotsize)
    if ax is None:
        ax = fig.add_subplot(111)
    ax = plot_wind_vectors(data_puffs, data_wind, t_val, ax, invert_colors=invert_colors, wind_vector=wind_vector)
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
    mlflow.log_metric("ppo/advantages_mean", advantages.mean().item(), step=j)
    mlflow.log_metric("ppo/advantages_std", advantages.std().item(), step=j)
    mlflow.log_metric("ppo/advantages_max", advantages.max().item(), step=j)
    mlflow.log_metric("ppo/advantages_min", advantages.min().item(), step=j)
    mlflow.log_metric("ppo/value_loss", value_loss, step=j)
    mlflow.log_metric("ppo/action_loss", action_loss, step=j)
    mlflow.log_metric("ppo/dist_entropy", dist_entropy, step=j)
    mlflow.log_metric("ppo/clip_fraction", clip_fraction, step=j)
    mlflow.log_metric("ppo/learning_rate", learning_rate, step=j)


def log_agent_learning_wind_obsver(j, advantages, value_loss, action_loss, dist_entropy, clip_fraction, learning_rate, aux_loss_dict, use_mlflow=True):
    log_agent_learning(j, advantages, value_loss, action_loss, dist_entropy, clip_fraction, learning_rate, use_mlflow=use_mlflow)
    all_wind_nll = aux_loss_dict['wind_nll_all']
    all_wind_sqerr = aux_loss_dict['wind_sqerr_all']
    all_wind_logvar = aux_loss_dict['wind_logvar_all']
    wind_nll_mean = all_wind_nll.mean().item()
    wind_nll_std  = all_wind_nll.std().item()
    wind_loss_epoch = aux_loss_dict["wind_loss_epoch"]
    mlflow.log_metric("wind_observer/wind_loss_mean", wind_loss_epoch, step=j)
    mlflow.log_metric("wind_observer/wind_nll_mean", wind_nll_mean, step=j)
    mlflow.log_metric("wind_observer/wind_nll_std",  wind_nll_std, step=j)
    mlflow.log_metric("wind_observer/wind_sqerr_mean", all_wind_sqerr.mean().item(), step=j)
    mlflow.log_metric("wind_observer/wind_logvar_mean", all_wind_logvar.mean().item(), step=j)


def log_curriculum_schedule(schedule_dict, j, use_mlflow=True):
    """Log current curriculum variable values at update index j"""
    if not use_mlflow:
        return

    for lesson_name, lesson_schedule in schedule_dict.items():
        # Find the most recent value at or before update j
        applicable_steps = [step for step in lesson_schedule.keys() if step <= j]
        if applicable_steps:
            current_step = max(applicable_steps)
            current_value = lesson_schedule[current_step]
            if isinstance(current_value, (list, tuple)):
                # e.g. {ds}_rotate_by holds an angle list ([0, 180] ... [0, 180, 90, -90, 45, -45, 135, -135]).
                # A list is not a plottable metric, so log how many options are unlocked instead.
                mlflow.log_metric(f"curriculum/{lesson_name}_n", len(current_value), step=j)
            else:
                mlflow.log_metric(f"curriculum/{lesson_name}", current_value, step=j)


def _log_eps_group(j, df, prefix):
    """Log per-update episode statistics for one group of episodes.

    Called once for the whole update and once per dataset, so the per-(dataset, update)
    thresholds used in the figure-2 panels can be rebuilt from wandb alone.
    """
    n = len(df)
    if n == 0:
        return
    mlflow.log_metric(f"{prefix}num_episodes", n, step=j)
    for outcome in ['HOME', 'OOB', 'OOT']:
        k = int((df['outcome'] == outcome).sum())
        mlflow.log_metric(f"{prefix}{outcome}_num", k, step=j)
        mlflow.log_metric(f"{prefix}{outcome}_ratio", k / n, step=j)

    # Realized rotate_by diversity: what was actually sampled, not what the schedule holds
    if 'rotate_by' in df.columns:
        mlflow.log_metric(f"{prefix}n_unique_rotate_by", int(df['rotate_by'].nunique()), step=j)

    init_dist = df['location_initial'].apply(np.linalg.norm)
    for name, s in [('init_distance', init_dist),
                    ('plume_density', df['plume_density'].astype(float)),
                    ('stray_initial', df['stray_initial'].astype(float))]:
        # mean + std + num_episodes is enough to recover the exact pooled mean/variance
        # across seeds later; the figure-2 density band is std**2.
        mlflow.log_metric(f"{prefix}{name}_mean", s.mean(), step=j)
        mlflow.log_metric(f"{prefix}{name}_std", s.std(ddof=1) if n > 1 else 0.0, step=j)
        mlflow.log_metric(f"{prefix}{name}_min", s.min(), step=j)
        mlflow.log_metric(f"{prefix}{name}_max", s.max(), step=j)


def log_eps_info(j, update_episodes_df, use_mlflow=True):
    if not use_mlflow or len(update_episodes_df) == 0:
        return
    # Legacy keys - kept verbatim so existing dashboards/queries keep working
    mlflow.log_metric("perf/init_distance", update_episodes_df['location_initial'].apply(np.linalg.norm).mean(), step=j)
    mlflow.log_metric("perf/stray_initial", update_episodes_df['stray_initial'].mean(), step=j)

    _log_eps_group(j, update_episodes_df, "perf/")
    for ds, group in update_episodes_df.groupby('dataset'):
        _log_eps_group(j, group, f"perf/{ds}/")


def log_update_wind_stats(j, wind_xy, use_mlflow=True, round_decimals=2):
    """Log per-update ambient-wind diversity for this run.

    Args:
        j (int): Update index (absolute, i.e. j_global).
        wind_xy: (num_steps * num_processes, 2) array of ``info['ambient_wind']``
            collected over the update - the wind the agents actually experienced,
            already rotated/mirrored by the env.
        round_decimals (int): Quantization of (direction, magnitude) before counting
            uniques. Matches the convention used to build the figure-2 wind panel.

    Note this counts uniques *within this run*; pooling across seeds is a mean of
    per-seed counts, not a set union.
    """
    if not use_mlflow or wind_xy is None or len(wind_xy) == 0:
        return
    wind_xy = np.asarray(wind_xy, dtype=np.float64)
    wind_dir = np.degrees(np.arctan2(wind_xy[:, 1], wind_xy[:, 0])) % 360.0
    wind_mag = np.hypot(wind_xy[:, 0], wind_xy[:, 1])

    # Quantize to integers and count unique (dir, mag) pairs through a single packed key.
    # ~2x faster than np.unique(..., axis=0), which lexsorts rows. Both quantities are
    # non-negative here, so a stride of max+1 packs them without collisions.
    scale = 10 ** round_decimals
    dir_i = np.rint(wind_dir * scale).astype(np.int64)
    mag_i = np.rint(wind_mag * scale).astype(np.int64)
    n_unique_vels = len(np.unique(dir_i * (mag_i.max() + 1) + mag_i))

    mlflow.log_metric("wind_stats/n_unique_wind_vels", n_unique_vels, step=j)
    mlflow.log_metric("wind_stats/n_unique_wind_dirs", len(np.unique(dir_i)), step=j)
    mlflow.log_metric("wind_stats/n_unique_wind_mags", len(np.unique(mag_i)), step=j)
    # Denominator: lets counts be normalized across runs with different num_processes/num_steps
    mlflow.log_metric("wind_stats/n_wind_samples", len(wind_xy), step=j)

def log_eps_artifacts(j, args, update_episodes_df, use_mlflow=True, log_artifacts=True, plot=True,
                      wind_xy=None):
    """
    Log episode statistics and plot a histogram of plume density for successful episodes.
    Just log episode statistics if not long_artifacts
    Args:
        j (int): Update index for logging and for labeling the plot.
        args (Namespace): Contains `save_dir` for saving the plot.
        update_episodes_df (pd.DataFrame): DataFrame with 'outcome', 'dataset', and 'plume_density'.
        wind_xy: optional (num_steps * num_processes, 2) array of ``info['ambient_wind']``
            collected over this update. See :func:`log_update_wind_stats`.
    """

    # Log episode statistics
    if not use_mlflow:
        return

    # Ambient wind is sampled every step regardless of whether any episode finished, so it
    # is logged before anything that keys off update_episodes_df (which may be empty).
    log_update_wind_stats(j, wind_xy)

    log_eps_info(j, update_episodes_df)

    if log_artifacts:
        log_path = f"{args.save_dir}/tmp/{args.model_fname.replace('.pt', '')}_eps_log_{j}.csv"
        update_episodes_df.to_csv(log_path, index=False)
        try:
            mlflow.log_artifact(log_path, artifact_path=f"eps_log", step=j)
        except Exception as e:
            print(f"Error logging artifact {log_path}: {e}")
    
    if plot:
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
                    mlflow.log_artifact(plt_path, artifact_path="figs/density", step=j)
                except Exception as e:
                    print(f"Error logging artifact {plt_path}: {e}")
        

# from a2c_ppo_acktr/storage.py
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])

"""Load a curriculum schedule saved by the Curriculum Scheduler (JSON) and
optionally realign it to a different total number of updates.

The scheduler saves a dict of the same shape produced by ``build_tc_schedule_dict``::

    {
        "birthx":    {"0": 0.7, "90": 0.60, ...},   # JSON makes the update keys strings
        "wind_cond": {"0": 1},
        ...
    }

``load_tc_schedule`` reads that file, converts the update keys back to ints, and
(optionally) rescales every lesson time so the curriculum spans a new number of
updates -- useful when you rerun training with a different ``total_number_periods``
than the one the curriculum was generated with.
"""

import json
import math


def realign_schedule(schedule, num_updates, orig_num_updates):
    """Rescale lesson times so the curriculum spans ``num_updates`` instead of
    ``orig_num_updates``.

    The new time for each lesson is ``old_time * (num_updates / orig_num_updates)``,
    rounded up (ceil) by default. Time 0 stays at 0, so the initial value of every
    track is preserved. Times are clamped to ``num_updates``; if two lessons land on
    the same update after rounding, the later (higher original time) one wins.

    Args:
        schedule: dict[str, dict[int, value]] with integer update keys.
        num_updates: target total number of updates.
        orig_num_updates: total number of updates used to generate the curriculum.
        round_up: ceil when True (default), otherwise round to nearest.

    Returns:
        A new dict[str, dict[int, value]] with realigned integer update keys.
    """
    if orig_num_updates == num_updates:
        # nothing to do (same horizon, or no reference to scale against)
        return {var: (dict(v) if isinstance(v, dict) else v)
                for var, v in schedule.items()}

    realign_ratio = num_updates / orig_num_updates
    _round = (lambda x: int(math.ceil(x)))

    realigned = {}
    for var, lessons in schedule.items():
        if not isinstance(lessons, dict):
            # scalar entries (e.g. restart_period) scale with the same ratio so the
            # number of restart cycles over the run stays the same
            realigned[var] = (min(_round(lessons * realign_ratio), num_updates)
                              if isinstance(lessons, (int, float)) else lessons)
            continue
        new_lessons = {}
        for t, v in sorted(lessons.items()):           # ascending -> last write wins on collision
            nt = min(_round(t * realign_ratio), num_updates)
            new_lessons[nt] = v
        realigned[var] = new_lessons
    return realigned, realign_ratio


def load_tc_schedule(path, num_updates):
    """Load a saved curriculum schedule JSON and optionally realign its update count.

    Args:
        path: path to the ``.json`` file saved by the scheduler.
        num_updates: target total number of updates. If None (default), the schedule
            is returned as saved, with no realignment.
        orig_num_updates: total number of updates the curriculum was *generated* with.
            If None, it is inferred from the largest lesson time in the schedule. Note
            that the last lesson usually sits below the true total (lessons are placed
            with ``endpoint=False``), so passing this explicitly is recommended for an
            exact rescale.
        round_up: passed through to :func:`realign_schedule`.

    Returns:
        schedule: dict[str, dict[int, value]] with integer update keys, realigned to
            ``num_updates`` when that differs from ``orig_num_updates``.
    """
    with open(path) as f:
        raw = json.load(f)

    # JSON object keys are strings -> turn the update indices back into ints.
    # Scalar top-level entries (e.g. restart_period) are kept as plain numbers.
    
    # grab and remove meta
    meta = raw.pop('meta', None)
    
    # convert update keys to ints, but only for dict entries (the lessons)
    schedule = {var: ({int(k): v for k, v in lessons.items()}
                      if isinstance(lessons, dict) else lessons)
                for var, lessons in raw.items()}

    orig_num_updates = meta.get('total_num_updates', None) if meta else None
    if orig_num_updates is not None:
        schedule, realign_ratio = realign_schedule(schedule, num_updates, orig_num_updates)
    restart_period = math.ceil(meta.get('restart_period', None) * realign_ratio) if meta else None

    return schedule, restart_period