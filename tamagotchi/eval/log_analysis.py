import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from moviepy.editor import ImageClip, concatenate_videoclips
from natsort import natsorted
import contextlib
import os
import tqdm
import sklearn.decomposition as skld
from tamagotchi import config
np.random.seed(config.seed_global)
from natsort import natsorted
import contextlib
import os
import tqdm
import glob 
import sys

ODOR_THRESHOLD = config.env['odor_threshold']

def vec2rad_norm_by_pi(x, y):
    """
    # https://physicsclassroom.com/mmedia/vectors/vd.cfm#:~:text=The%20convention%20upon%20which%20we,of%20rotation%20from%20due%20east.
    Standard CCW notation, centered at 0
    Returns +1 is +180-deg, and -0.999 is -180-deg 
    vec2rad_norm_by_pi(0,0) # 0.0
    vec2rad_norm_by_pi(1,0) # 0.0
    vec2rad_norm_by_pi(1,1) # 0.25
    vec2rad_norm_by_pi(0,1) # 0.5
    vec2rad_norm_by_pi(-1,1) # 0.75
    vec2rad_norm_by_pi(-1,0) # 1.0
    vec2rad_norm_by_pi(-1,-0.01) # -0.997 
    vec2rad_norm_by_pi(-1,-1) # -0.75
    vec2rad_norm_by_pi(0,-1) # -0.5
    vec2rad_norm_by_pi(+1,-1) # -0.25
    """
    return np.angle( x + 1j*y, deg=False )/np.pi # note div by np.pi!

def rad_over_pi_shift2_01(theta):
    """
    input: between -1 (-180-deg) and +1 (+180 deg)
    output: between 0 (-180-deg) and +1 (+180 deg)

    """
    return (theta+1)/2

# TODO: Stack Overflow citation missing!
def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])


def rescale_col(series, vmax=None):
    s = series - series.min()
    if vmax is not None:
        s /= vmax
    else:
        s /= s.max()
    return s


def get_selected_df(model_dir, use_datasets, 
    n_episodes_home=240, 
    n_episodes_other=240, 
    min_ep_steps=0, 
    oob_only=True,
    balanced=True, 
    verbose=False, 
    log_fname=None):
    episodes_df = []
    for dataset in use_datasets:
        if not log_fname:
            log_fname = f'{model_dir}/{dataset}.pkl'
        with open(log_fname, 'rb') as f_handle:
            episode_logs = pickle.load(f_handle)

        for idx in range(len(episode_logs)):
            log = episode_logs[idx]
            episodes_df.append({'dataset': dataset, 
                                'idx': idx,
                                'ep_length': len(log['trajectory']),
                                'log': log,
                                'outcome': log['infos'][-1][0]['done'],
                               })

    episodes_df = pd.DataFrame(episodes_df).sort_values(by='ep_length', ascending=False)
    if verbose:
        print(f"Found {episodes_df.groupby(['dataset', 'outcome']).count()} in {log_fname}, after filter by min {min_ep_steps} steps")
    episodes_df = episodes_df.query("ep_length >= @min_ep_steps")
    if oob_only:
        selected_df = pd.concat([ 
            episodes_df.query('outcome == "HOME"').groupby('dataset').head(n_episodes_home),
            episodes_df.query('outcome == "OOB"').groupby('dataset').head(n_episodes_other),
            pd.DataFrame({})
        ]).reset_index(drop=True)
    else:
        selected_df = pd.concat([ 
            episodes_df.query('outcome == "HOME"').groupby('dataset').head(n_episodes_home),
            episodes_df.query('outcome != "HOME"').groupby('dataset').head(n_episodes_other),
            pd.DataFrame({})
        ]).reset_index(drop=True)
    if verbose:
        print(f"Found {episodes_df.groupby(['dataset', 'outcome']).count()} in {log_fname}, after selecting specific number of episodes")

    if balanced:
        counts_df = selected_df.groupby(['dataset', 'outcome']).count()
        counts_df

        # Balance out HOME and OOB in selected_df
        balanced_df = []
        for dataset in use_datasets:
            min_count = counts_df.query("dataset == @dataset")['idx'].min()
            balanced_df.append( selected_df.query("dataset == @dataset").groupby('outcome').head(min_count) )

        balanced_df = pd.concat(balanced_df)
        balanced_df.groupby(['dataset', 'outcome']).count()
        selected_df = balanced_df
        if verbose:
            print(f"Found {episodes_df.groupby(['dataset', 'outcome']).count()} in {log_fname}, after balancing")
        
    if verbose:
        print("model_dir", model_dir)
        logfiles = natsorted(glob.glob("/".join([model_dir, '*.pkl'])))
        model_seed = model_dir.rstrip('/').split('/')[-1].split('_')[1]
        print("model_seed ---->", model_seed)
        print(f"Found {len(logfiles)} .pkl evaluation logs in {model_dir}")
        print(f"selected N eps {selected_df.shape}")
        print(f"Final episode breakdown: \n {selected_df.groupby(['dataset', 'outcome']).count()}")
        
    return(selected_df)

def get_pca_common(selected_df, n_comp = 12, is_recurrent=True):
    h_episodes = []
    traj_dfs = []
    # squash_action = True

    for episode_log in selected_df['log']:
        ep_activity = get_activity(episode_log, is_recurrent, do_plot=False)
        h_episodes.append(ep_activity)

    h_episodes_stacked = np.vstack(h_episodes)
    # print(h_episodes_stacked.shape)

    pca_common = skld.PCA(n_comp, whiten=False)
    # pca_common = skld.PCA(whiten=False)
    pca_common.fit(h_episodes_stacked)
    return pca_common

def regime_to_colors(regime_list):
    colors = [ config.regime_colormap[x] for x in regime_list ]
    return colors

def get_regimes(traj_df, outcome, RECOVER_MIN=12, RECOVER_MAX=25, seed=None):
    # SEGMENT/label trajectory timesteps: WARMUP, SEARCH, TRACK, RECOVER    
    # TRACK: Experienced odor within RECOVER_MAX steps
    # RECOVER: 

    if seed is not None and seed in config.seedmeta.keys():
        RECOVER_MIN = config.seedmeta[seed]['recover_min']
        RECOVER_MAX = config.seedmeta[seed]['recover_max']
        print("seed specific thresholds", RECOVER_MIN, RECOVER_MAX)

    traj_df['regime'] = 'RECOVER'

    # TRACK
    traj_df.loc[traj_df['odor_lastenc'] <= RECOVER_MIN, 'regime'] = 'TRACK'

    # SEARCH
    traj_df.loc[traj_df['odor_lastenc'] >= RECOVER_MAX, 'regime'] = 'SEARCH'

    # RECOVER: Between RECOVER_MIN and RECOVER_MAX
    recover_idxs = traj_df['odor_lastenc'].apply(lambda x: x in np.arange(RECOVER_MIN, RECOVER_MAX))
    traj_df.loc[recover_idxs, 'regime'] = 'RECOVER'

    # Warm-up in RECOVER
    traj_df.iloc[:RECOVER_MIN, traj_df.columns.get_loc('regime')] = 'RECOVER'


    return traj_df['regime']


# mark when the wind regime changes
def get_wind_change_regimes(traj_df, wind_change_frame_threshold=5, frame_rate=0.04, verbose=False):
    threshold = wind_change_frame_threshold * frame_rate
    traj_df['wind_regime'] = 'tracking'
    # wind just changed within the last N frames - anemometric behavior should follow 
    traj_df['wind_regime'].loc[ traj_df['time_since_last_wind_change'] <= threshold ] = 'anemometric'
    if verbose:
        print("Annotation of wind regimes:")
        print(f"{traj_df['wind_regime'].value_counts()}")
        print(f"Threshold for wind change regime is {threshold} seconds \n")
        

def get_eval_dfs_and_stack_them(model_fname, use_datasets, number_of_eps, exp_dir='eval', full_model_dir=None, verbose=False, oob_only=True, balanced=False):
    # used when visualizing trajectories and actions taken
    is_recurrent = True
    # load eval episodes from pkl files
    model_dir = model_fname.replace('.pt', '/').replace("weights", exp_dir) 
    if full_model_dir is not None:
        model_dir = full_model_dir
        
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} does not exist")
        sys.exit(0)
    # read pkl file into PD dataframe - each row is an epoch
    selected_df = get_selected_df(model_dir, 
                                    [use_datasets], 
                                    n_episodes_home=number_of_eps, 
                                    n_episodes_other=number_of_eps,
                                    balanced=balanced,
                                    oob_only=oob_only,
                                    min_ep_steps=0, 
                                    verbose=verbose)
    if verbose:
        print("model_dir", model_dir)
        logfiles = natsorted(glob.glob(model_dir + '*.pkl'))
        model_seed = model_dir.rstrip('/').split('/')[-1].split('_')[1]
        print("model_seed ---->", model_seed)
        print(f"Found {len(logfiles)} .pkl evaluation logs in {model_dir}")
        print(f"selected N eps {selected_df.shape}")
        print(f"Episode breakdown: \n {selected_df.groupby(['dataset', 'outcome']).count()}")

    # get traj data and stack them
    traj_dfs = []
    squash_action = True
    # for episode_log in tqdm.tqdm(selected_df['log']):
    for idx,row in tqdm.tqdm(selected_df.iterrows()):
        dataset = row['dataset']
        episode_log = row['log']
        # use get_traj_df_tmp to calculate head direction
        traj_df = get_traj_df_tmp(episode_log, 
                                            extended_metadata=True, 
                                            squash_action=squash_action)
        traj_df['idx'] = np.arange(traj_df.shape[0], dtype=int)
        traj_df['ep_idx'] = row['idx']
        traj_df['dataset'] = dataset
        traj_df['outcome'] = row['outcome']
        traj_df['loc_x_dt'] = traj_df['loc_x'].diff() 
        traj_df['loc_y_dt'] = traj_df['loc_y'].diff() 
        traj_dfs.append(traj_df)

    traj_df_stacked = pd.concat(traj_dfs, ignore_index=True)
    if verbose:
        print(f"Stacked traj dfs shape: {traj_df_stacked.shape}")
    return traj_df_stacked


def get_traj_and_activity_and_stack_them(eval_log_pkl_df: pd.DataFrame, 
                                         obtain_neural_activity: bool = True, 
                                         obtain_traj_df: bool = True, 
                                         get_traj_tmp: bool = True,
                                         extended_metadata: bool = False) -> pd.DataFrame:
    """
    Load and stack trajectory and neural activity data from evaluation logs. 

    Args:
        eval_log_pkl_df (pd.DataFrame): DataFrame containing evaluation logs.
        obtain_neural_activity: Flag to obtain neural activity data. Default is True.
        obtain_traj_df: Flag to obtain trajectory DataFrame. Default is True.
        get_traj_tmp: Flag to use get_traj_df_tmp to calculate head direction and course direction. Default is True.
            Note: tmp calculates from info which is unnormalized. Else get from obs which can either be raw or normalized. If vecNormalize is saved, then obs is raw, which can be normalized later.
                TODO: make sure that obs is always normalized - keep a copy of the raw version in info. This way open loop is easier with the normalized obs
    Returns:
        pd.DataFrame: Stacked trajectory DataFrame.
        np.ndarray: Stacked neural activity data.
        'DUMMY' if df was not to be obtained.
    """
    
    # load and stack data if not provided
    if obtain_neural_activity or obtain_traj_df:    
        is_recurrent = True
        squash_action = True
        h_episodes = []
        traj_dfs = []
        for idx, row  in tqdm.tqdm(eval_log_pkl_df.iterrows()):
            episode_log = row['log']
            if obtain_neural_activity:
                ep_neural_activity = get_activity(episode_log, is_recurrent, do_plot=False)
                h_episodes.append(ep_neural_activity)
            if obtain_traj_df:
                if get_traj_tmp:
                    traj_df = get_traj_df_tmp(episode_log, 
                                                extended_metadata=extended_metadata, 
                                                squash_action=squash_action)
                else:
                    traj_df = get_traj_df(episode_log, 
                                            extended_metadata=extended_metadata, 
                                            squash_action=squash_action)
                traj_df['tidx'] = np.arange(traj_df.shape[0], dtype=int)
                for colname in ['dataset', 'idx', 'outcome']:
                    traj_df[colname] = row[colname] 
                    if colname == 'idx':
                        traj_df['ep_idx'] = row[colname]
                traj_dfs.append(traj_df)
        stacked_neural_activity = stacked_traj_df = 'DUMMY'
        if obtain_neural_activity:
            stacked_neural_activity = np.vstack(h_episodes)
        if obtain_traj_df:
            stacked_traj_df = pd.concat(traj_dfs)

    return stacked_traj_df, stacked_neural_activity


def calc_time_since_last_wind_change(eps_df):
    """
    Calculates the time since the last change in wind direction.

    Parameters:
    - eps_df (pandas.DataFrame): The input DataFrame of eval trajectories for a single episode, containing extended metadata.s

    Returns:
    - pandas.DataFrame: The input DataFrame with an additional column 'time_since_last_wind_change'
                        representing the time since the last change in wind direction.

    Raises:
    - AssertionError: If the time interval between consecutive 't_val' values is not constant.

    """
    # sanity check - delta on t_val should be the same 
    t_val_dt = eps_df['t_val'].diff().dropna().unique()
    t_val_dt = set(np.round(t_val_dt, 2))
    assert len(t_val_dt) == 1, f"t_val_dt is not constant: {t_val_dt}"
    t_val_dt = t_val_dt.pop()
    # first row is always nan - fill with 0
    eps_df['wind_angle_ground_theta_dt1'].iloc[0] = 0
    # calculate time since last change in wind direction 
    time_since_last_wind_change = []
    time_count = 0
    for idx, row in eps_df.iterrows():
        time_count += t_val_dt
        if row['wind_angle_ground_theta_dt1']: # delta theta is not 0 - wind just changed
            time_count = 0    
        time_since_last_wind_change.append(time_count)
    
    eps_df = eps_df.assign(time_since_last_wind_change = time_since_last_wind_change)
    return eps_df

def get_traj_df(episode_log, 
    extended_metadata: bool = False, 
    squash_action: bool = False,
    n_history: int = 20,
    seed: int = None,
    ) -> pd.DataFrame:
    '''
    Generate a trajectory DataFrame from an episode log.
    The following variables are extracted from obs. Only should do this when obs is not normalized.
        wind_theta_obs
        agent_angle_theta
        ego_course_direction_theta
        
        Note: normalized inputs may be relevant for fitting LDS. Leave obs as normalized and read from infos in future. 

    Parameters:
    - episode_log (dict): The episode log containing trajectory data.
    - extended_metadata (bool): Flag to include extended metadata in the DataFrame. Default is False.
    - squash_action (bool): Flag to squash the action values. Default is False.
    - n_history (int): Maxinum number of history steps to include when calculating n-step ENV/EWA/MA odor. Creates from 2~n_history info. Default is 20. 
    - seed (int): Seed value for random number generation. Used in get regimes Default is None.

    Returns:
    - traj_df (pd.DataFrame): The generated trajectory DataFrame.
    '''

    # squash_action=True only needed for old log files 
    # (this is now done in evalCli itself, during creation of episode_log)

    # Basic trajectory (x, y)
    trajectory = episode_log['trajectory']
    traj_df = pd.DataFrame(trajectory)  
    traj_df.columns = ['loc_x', 'loc_y']   

    # time
    traj_df['t_val'] = [record[0]['t_val'] for record in episode_log['infos']]

    # Observations & Actions
    obs = [x[0] for x in episode_log['observations']]
    obs = pd.DataFrame(obs)
    
    if obs.shape[1] == 3:
        obs = obs.iloc[:, -3:] # handles STACKING > 0 # keep line from sat's code. Not sure why this is here
        obs.columns = ['wind_x', 'wind_y', 'odor']
    elif obs.shape[1] == 7:
        obs =  obs.iloc[:, -7:] # obs in PEv3 has 7 columns - works as expected # this are normalized observations
        obs.columns = ['wind_x', 'wind_y', 'odor', 'agent_angle_x', 'agent_angle_y', 'ego_course_direction_x', 'ego_course_direction_y']
    
    # write wind observation into df
    obs['wind_theta_obs'] = obs.apply(lambda row: vec2rad_norm_by_pi(row['wind_x'], row['wind_y']), axis=1)
    traj_df['wind_theta_obs'] = rad_over_pi_shift2_01(obs['wind_theta_obs'])
    traj_df['wind_x_obs'] = obs['wind_x']
    traj_df['wind_y_obs'] = obs['wind_y']
    # write agent angle observation into df
    agent_angle_theta = rad_over_pi_shift2_01(obs.apply(lambda row: vec2rad_norm_by_pi(row['agent_angle_x'], row['agent_angle_y']), axis=1))
    traj_df['agent_angle_x'] = obs['agent_angle_x']
    traj_df['agent_angle_y'] = obs['agent_angle_y']
    traj_df['agent_angle_theta'] = agent_angle_theta
    # get from info, instead of obs
    traj_df['agent_angle_ground_theta'] = [ rad_over_pi_shift2_01( 
        vec2rad_norm_by_pi(record[0]['angle'][0], record[0]['angle'][1]) ) \
        for record in episode_log['infos']]
    # write course direction observation into df
    ego_course_direction_theta = rad_over_pi_shift2_01(obs.apply(lambda row: vec2rad_norm_by_pi(row['ego_course_direction_x'], row['ego_course_direction_y']), axis=1))
    traj_df['ego_course_direction_x'] = obs['ego_course_direction_x']
    traj_df['ego_course_direction_y'] = obs['ego_course_direction_y']
    traj_df['ego_course_direction_theta'] = ego_course_direction_theta
    # get true wind direction from info
    if obs.shape[1] == 7: 
        true_wind_direction_key = 'ambient_wind'
    else: # for relative wind agents - still consider true wind direction 
        # get true wind info for action dist. around wind changes 
        true_wind_direction_key = 'wind_ground'
    traj_df['wind_angle_ground_theta'] = [ rad_over_pi_shift2_01( 
        vec2rad_norm_by_pi(record[0][true_wind_direction_key][0], record[0][true_wind_direction_key][1]) ) for record in episode_log['infos']]
    traj_df['wind_angle_ground_x'] = [ record[0][true_wind_direction_key][0] for record in episode_log['infos']]
    traj_df['wind_angle_ground_y'] = [ record[0][true_wind_direction_key][1] for record in episode_log['infos']]
    traj_df['wind_speed_ground'] = [ np.linalg.norm(record[0][true_wind_direction_key]) for record in episode_log['infos']]

    act = episode_log['actions'] 
    act = pd.DataFrame(act)
    if squash_action:
        act = (np.tanh(act) + 1)/2
    act.columns = ['step', 'turn']
    traj_df['step'] = act['step']
    traj_df['turn'] = act['turn']

    traj_df['odor_raw'] = obs['odor'] # added for open loop perturbation analysis - do not rectify
    traj_df['odor_eps_log'] = episode_log['odor'] # added for visualization - after sensing
    traj_df['odor_obs'] = [0. if x <= config.env['odor_threshold'] else x for x in traj_df['odor_raw']]

    traj_df['stray_distance'] = [record[0]['stray_distance'] for record in episode_log['infos']]

    # Observation derived
    traj_df['odor_01'] = [0 if x <= config.env['odor_threshold'] else 1 for x in traj_df['odor_eps_log']]
    traj_df['odor_clip'] = traj_df['odor_obs'].clip(lower=0., upper=1.0)

    # time since last encounter
    def _count_lenc(lenc0, maxcount=None):
        count = 0
        lenc = [0]
        for i in range(len(lenc0) - 1):
            count = 0 if lenc0[i] == 0 else count+1
            if maxcount is not None:
                count = maxcount if count >= maxcount else count
            lenc.append(count)
        return lenc
    # traj_df['odor_lastenc'] = _count_lenc( 1 - traj_df['odor_01'], maxcount=15 )
    traj_df['odor_lastenc'] = _count_lenc( 1 - traj_df['odor_01'], maxcount=None )

    ### REGIMEs
    outcome = episode_log['infos'][-1][0]['done'] 
    traj_df['regime'] = get_regimes(traj_df, outcome, seed=seed)
    ##

    if extended_metadata:
        traj_df['t_val_norm'] = rescale_col(traj_df['t_val'], vmax=12) # 300/25 steps/fps

        # Observation derived
        # traj_df['odor_cummax'] = traj_df['odor_obs'].cummax()
        # traj_df['odor_cummax'] = rescale_col(traj_df['odor_cummax']) 

        # Add a range of ENC (encounters)
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_enc_{j}'
            traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).ewm(span=j).mean()*25
            # traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).rolling(j).mean()*25
            # traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).ewm(span=j).mean()*25

        # Add a range of EWM
        traj_df['odor_ewm'] = traj_df['odor_clip'].ewm(span=15).mean()
        # traj_df['odor_ewm'] = rescale_col(traj_df['odor_ewm']) 
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_ewm_{j}'
            traj_df[colname] = traj_df['odor_clip'].ewm(span=j).mean()
            # traj_df[colname] = traj_df['odor_clip'].ewm(span=j).mean()
            # traj_df[colname] = traj_df['odor_01'].ewm(span=j).mean()
            # traj_df[colname + '_norm'] = rescale_col(traj_df[colname]) 

        # Add a range of MA
        traj_df['odor_ma'] = traj_df['odor_clip'].rolling(15).mean()
        # traj_df['odor_ma_norm'] = rescale_col(traj_df['odor_ma']) 
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_ma_{j}'
            traj_df[colname] = traj_df['odor_clip'].rolling(j).mean()
            # traj_df[colname] = traj_df['odor_01'].rolling(j).mean()
            # traj_df[colname + '_norm'] = rescale_col(traj_df[colname]) 

        traj_df['odor_lastenc_norm'] = rescale_col(traj_df['odor_lastenc']) 

        # Location derived
        traj_df['radius'] = [np.linalg.norm(record[0]['location']) for record in episode_log['infos']]
        # traj_df['radius_norm'] = rescale_col(traj_df['radius'], vmax=12) # Max dist + max stray

        # traj_df['stray_distance_norm'] = rescale_col(traj_df['stray_distance']) # Max stray distance

        traj_df['r_step'] = [record[0]['r_radial_step'] for record in episode_log['infos']]
        # traj_df['r_step_norm'] = rescale_col(traj_df['r_step']) 

        # Differences
        colnames_diff = [
            'step',
            # 'turn', # turn 0.5 means no turning 
            'loc_x', 
            'loc_y', 
            'wind_theta_obs', # wind input to the model... depends on the experiment
            'odor_obs', 
            'odor_01',
            'odor_clip', 
            'odor_lastenc', 
            'radius', 
            'stray_distance',
            'r_step', 
            # 'agent_angle_ground_theta', # discontinuous - not useful for diff
            'wind_speed_ground',
            'wind_angle_ground_theta' # discontinuous, but ok since limited to first/fourth quadrant
            ] 
        if obs.shape[1] == 7:
            colnames_diff.append('ego_course_direction_theta')
        for col in colnames_diff:
            traj_df[f'{col}_dt1'] = traj_df[col].diff()
            # traj_df[f'{col}_dt2'] = traj_df[f'{col}_dt1'].diff()

    return traj_df

def get_traj_df_tmp(episode_log, 
    extended_metadata: bool = False, 
    squash_action: bool = False,
    n_history: int = 20,
    seed: int = None,
    ) -> pd.DataFrame:
    '''
    A temporary version that calculates ego_course_direction from info instead of using from observations, and gets agent_angle from info instead of observations. 
    This is used for agent 951 analyses because the sensory observations have been normalized by the VecNormalize filter. Need to plot with the raw observations which were saved before normalization.
    
    Generate a trajectory DataFrame from an episode log.

    Parameters:
    - episode_log (dict): The episode log containing trajectory data.
    - extended_metadata (bool): Flag to include extended metadata in the DataFrame. Default is False.
    - squash_action (bool): Flag to squash the action values. Default is False.
    - n_history (int): Maxinum number of history steps to include when calculating n-step ENV/EWA/MA odor. Creates from 2~n_history info. Default is 20. 
    - seed (int): Seed value for random number generation. Used in get regimes Default is None.

    Returns:
    - traj_df (pd.DataFrame): The generated trajectory DataFrame.
    '''

    # squash_action=True only needed for old log files 
    # (this is now done in evalCli itself, during creation of episode_log)

    # Basic trajectory (x, y)
    trajectory = episode_log['trajectory']
    traj_df = pd.DataFrame(trajectory)  
    traj_df.columns = ['loc_x', 'loc_y']   

    # time
    traj_df['t_val'] = [record[0]['t_val'] for record in episode_log['infos']]

    # Observations & Actions
    obs = [x[0] for x in episode_log['observations']]
    obs = pd.DataFrame(obs)
    
    if obs.shape[1] == 3:
        obs = obs.iloc[:, -3:] # handles STACKING > 0 # keep line from sat's code. Not sure why this is here
        obs.columns = ['wind_x', 'wind_y', 'odor']
    elif obs.shape[1] == 7:
        obs =  obs.iloc[:, -7:] # obs in PEv3 has 7 columns - works as expected # this are normalized observations
        obs.columns = ['wind_x', 'wind_y', 'odor', 'agent_angle_x', 'agent_angle_y', 'ego_course_direction_x', 'ego_course_direction_y']
    elif obs.shape[1] == 9:
        obs = obs.iloc[:, -9:] # haltere 
        obs.columns = ['wind_x', 'wind_y', 'odor', 'agent_angle_x', 'agent_angle_y', 'ego_course_direction_x', 'ego_course_direction_y', 'haltere_air_acc', 'haltere_ang_acc']
    
    # write wind observation into df
    traj_df['wind_theta_obs'] = obs.apply(lambda row: vec2rad_norm_by_pi(row['wind_x'], row['wind_y']), axis=1)
    traj_df['wind_theta_obs'] = rad_over_pi_shift2_01(traj_df['wind_theta_obs'])
    traj_df['wind_x_obs'] = obs['wind_x']
    traj_df['wind_y_obs'] = obs['wind_y']
    # calc agent angle observation from info 
    traj_df['agent_angle_x'] = [ record[0]['angle'][0] for record in episode_log['infos']]
    traj_df['agent_angle_y'] = [ record[0]['angle'][1] for record in episode_log['infos']]
    traj_df['agent_angle_ground_theta'] = [ rad_over_pi_shift2_01( 
        vec2rad_norm_by_pi(record[0]['angle'][0], record[0]['angle'][1]) ) \
        for record in episode_log['infos']]
    if obs.shape[1] == 7: # only consider these info if obs has visual feedback - earlier trials logs were different
        # calculate course direction from info
        allo_ground_velocity  = [record[0]['ground_velocity'] for record in episode_log['infos']]
        # same calc as vec2rad_norm_by_pi, except do not normalize by pi
        allocentric_course_direction_radian = [np.angle(gv[0] + 1j*gv[1], deg=False) for gv in allo_ground_velocity]
        allocentric_head_direction_radian = [np.angle(record[0]['angle'][0] + 1j*record[0]['angle'][1], deg=False) for record in episode_log['infos']] 
        egocentric_course_direction_radian = np.array(allocentric_course_direction_radian) - np.array(allocentric_head_direction_radian) # leftward positive - standard CWW convention
        ego_course_direction_x, ego_course_direction_y = np.cos(egocentric_course_direction_radian), np.sin(egocentric_course_direction_radian)
        egocentric_course_direction_theta = rad_over_pi_shift2_01(vec2rad_norm_by_pi(ego_course_direction_x, ego_course_direction_y)) # normalize by pi and then shift to 0-1
        traj_df['ego_course_direction_x'] = ego_course_direction_x
        traj_df['ego_course_direction_y'] = ego_course_direction_y
        traj_df['ego_course_direction_theta'] = egocentric_course_direction_theta
        traj_df['allo_ground_velocity'] = allo_ground_velocity
    # get true wind direction from info
    if obs.shape[1] == 7 or obs.shape[1] == 9: 
        true_wind_direction_key = 'ambient_wind'
    else: # for relative wind agents - still consider true wind direction 
        # get true wind info for action dist. around wind changes 
        true_wind_direction_key = 'wind_ground'
    traj_df['wind_angle_ground_theta'] = [ rad_over_pi_shift2_01( 
        vec2rad_norm_by_pi(record[0][true_wind_direction_key][0], record[0][true_wind_direction_key][1]) ) for record in episode_log['infos']]
    traj_df['wind_angle_ground_x'] = [ record[0][true_wind_direction_key][0] for record in episode_log['infos']]
    traj_df['wind_angle_ground_y'] = [ record[0][true_wind_direction_key][1] for record in episode_log['infos']]
    traj_df['wind_speed_ground'] = [ np.linalg.norm(record[0][true_wind_direction_key]) for record in episode_log['infos']]
    # for rel wind backwrad compatibility
    if 'air_velocity' in episode_log['infos'][0][0].keys():
        traj_df['air_velocity'] = [ record[0]['air_velcity'] for record in episode_log['infos']]
    # else:
        # print("air_velocity not found in episode_log['infos'][0][0].keys() Could be looking at an older log file. May or may not be a problem.")
    act = episode_log['actions'] 
    act = pd.DataFrame(act)
    if squash_action:
        act = (np.tanh(act) + 1)/2
    act.columns = ['step', 'turn']
    traj_df['step'] = act['step']
    traj_df['turn'] = act['turn']
    traj_df['stray_distance'] = [record[0]['stray_distance'] for record in episode_log['infos']]
    # Observation derived
    traj_df['odor_raw'] = obs['odor'] # added for open loop perturbation analysis - do not rectify
    traj_df['odor_obs'] = [0. if x <= config.env['odor_threshold'] else x for x in traj_df['odor_raw']]
    traj_df['odor_clip'] = traj_df['odor_obs'].clip(lower=0., upper=1.0)
    # for rel wind backwrad compatibility
    if 'odor_obs' in episode_log['infos'][0][0].keys():
        traj_df['odor_eps_log'] = [ record[0]['odor_obs'] for record in episode_log['infos']]# added for visualization - after sensing
        traj_df['odor_01'] = [0 if x <= config.env['odor_threshold'] else 1 for x in traj_df['odor_eps_log']]
    else:
        traj_df['odor_01'] = [0 if x <= config.env['odor_threshold'] else 1 for x in traj_df['odor_obs']]
        # print("[NOTE] odor_obs not found in episode_log['infos']. Could be an older file. Odor_01 depends on RMS normalized odor, may not be accurate.")
        
    # time since last encounter
    def _count_lenc(lenc0, maxcount=None):
        count = 0
        lenc = [0]
        for i in range(len(lenc0) - 1):
            count = 0 if lenc0[i] == 0 else count+1
            if maxcount is not None:
                count = maxcount if count >= maxcount else count
            lenc.append(count)
        return lenc
    # traj_df['odor_lastenc'] = _count_lenc( 1 - traj_df['odor_01'], maxcount=15 )
    traj_df['odor_lastenc'] = _count_lenc( 1 - traj_df['odor_01'], maxcount=None )

    ### REGIMEs
    outcome = episode_log['infos'][-1][0]['done'] 
    traj_df['regime'] = get_regimes(traj_df, outcome, seed=seed)
    ###

    if extended_metadata:
        traj_df['t_val_norm'] = rescale_col(traj_df['t_val'], vmax=12) # 300/25 steps/fps

        # Observation derived
        # traj_df['odor_cummax'] = traj_df['odor_obs'].cummax()
        # traj_df['odor_cummax'] = rescale_col(traj_df['odor_cummax']) 

        # Add a range of ENC (encounters)
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_enc_{j}'
            traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).ewm(span=j).mean()*25
            # traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).rolling(j).mean()*25
            # traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).ewm(span=j).mean()*25

        # Add a range of EWM
        traj_df['odor_ewm'] = traj_df['odor_clip'].ewm(span=15).mean()
        # traj_df['odor_ewm'] = rescale_col(traj_df['odor_ewm']) 
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_ewm_{j}'
            traj_df[colname] = traj_df['odor_clip'].ewm(span=j).mean()
            # traj_df[colname] = traj_df['odor_clip'].ewm(span=j).mean()
            # traj_df[colname] = traj_df['odor_01'].ewm(span=j).mean()
            # traj_df[colname + '_norm'] = rescale_col(traj_df[colname]) 

        # Add a range of MA
        traj_df['odor_ma'] = traj_df['odor_clip'].rolling(15).mean()
        # traj_df['odor_ma_norm'] = rescale_col(traj_df['odor_ma']) 
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_ma_{j}'
            traj_df[colname] = traj_df['odor_clip'].rolling(j).mean()
            # traj_df[colname] = traj_df['odor_01'].rolling(j).mean()
            # traj_df[colname + '_norm'] = rescale_col(traj_df[colname]) 
        
        # Add a range of MA wind direction - wind doesn't change often enough for this to be interesting
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'wind_angle_ground_theta_ma_{j}'
            traj_df[colname] = traj_df['wind_angle_ground_theta'].rolling(j).mean()

        traj_df['odor_lastenc_norm'] = rescale_col(traj_df['odor_lastenc']) 

        # Location derived
        traj_df['radius'] = [np.linalg.norm(record[0]['location']) for record in episode_log['infos']]
        # traj_df['radius_norm'] = rescale_col(traj_df['radius'], vmax=12) # Max dist + max stray

        # traj_df['stray_distance_norm'] = rescale_col(traj_df['stray_distance']) # Max stray distance

        traj_df['r_step'] = [record[0]['r_radial_step'] for record in episode_log['infos']]
        # traj_df['r_step_norm'] = rescale_col(traj_df['r_step']) 

        # Differences
        colnames_diff = [
            'step',
            # 'turn', # turn 0.5 means no turning 
            'loc_x', 
            'loc_y', 
            'wind_theta_obs', # wind input to the model... depends on the experiment
            'odor_obs', 
            'odor_01',
            'odor_clip', 
            'odor_lastenc', 
            'radius', 
            'stray_distance',
            'r_step', 
            # 'agent_angle_ground_theta', # discontinuous - not useful for diff
            'wind_speed_ground',
            'wind_angle_ground_theta' # discontinuous, but ok since limited to first/fourth quadrant
            ] 
        if obs.shape[1] == 7:
            colnames_diff.append('ego_course_direction_theta')
        for col in colnames_diff:
            traj_df[f'{col}_dt1'] = traj_df[col].diff()
            # traj_df[f'{col}_dt2'] = traj_df[f'{col}_dt1'].diff()

    return traj_df


# Trajectory DF --> Episode DF row
def get_episode_metadata(log, odor_threshold=ODOR_THRESHOLD, squash_action=False):    
    # Observation and Action information
    traj_df = get_traj_df(log, extended_metadata=False, squash_action=squash_action)
    ep_length = len(traj_df)
    
    turn_score = np.mean(np.abs(traj_df['turn'] - 0.5)) # More for more turns
    speed_score = np.mean(traj_df['step'])
    
    off_plume = traj_df['odor_obs'] < odor_threshold
    off_plume_fraction = 0 if ep_length==0 else np.sum(off_plume+0)/ep_length

    exit_rle = rle(off_plume+0)
    num_exits = np.sum(exit_rle[2] == 1)  # Num of exits from plume
    exit_durations = exit_rle[0][ exit_rle[2] == 1 ] # Lengths for when below threshold
    max_exit_duration = 0 if len(exit_durations)==0 else np.max(exit_durations)

    r_counts = traj_df['regime'].value_counts()
    for key in ['TRACK', 'RECOVER', 'SEARCH']:
        if key not in r_counts.keys():
            r_counts[key] = 0
    n_track = r_counts['TRACK']
    n_recover = r_counts['RECOVER']
    n_search = r_counts['SEARCH']

    # Other episode information
    infos = log['infos'][-1][0]
    
    return {
        'done': infos['done'],
        'ep_length': ep_length,
        'n_track': n_track,
        'n_recover': n_recover,
        'n_search': n_search,
        'off_plume_fraction': off_plume_fraction,
        'turn_score': turn_score,
        'speed_score': speed_score,
        'num_exits': num_exits,
        'max_exit_duration': max_exit_duration,

        'max_stray_distance': np.max(traj_df['stray_distance']),
        'max_odor_lastenc': np.max(traj_df['odor_lastenc']),
        'avg_stray_distance': np.mean(traj_df['stray_distance']),
        'avg_odor_lastenc': np.mean(traj_df['odor_lastenc']),
        'median_stray_distance': np.median(traj_df['stray_distance']),
        'median_odor_lastenc': np.median(traj_df['odor_lastenc']),

        'agent_angle_ground_median': np.median(traj_df['agent_angle_ground']),

        'radius_final': np.linalg.norm(infos['location']),
        'radius_covered': np.linalg.norm(infos['location_initial']) - np.linalg.norm(infos['location']),
        'end_x': infos['location'][0],
        'end_y': infos['location'][1],
        'start_x': infos['location_initial'][0],
        'start_y': infos['location_initial'][1],
        'reward': infos['reward'],
    }




def get_activity(log, is_recurrent, do_plot=False):
    '''
        This function retrieves the neural activity from the log data.
        
        Parameters:
        - log: The log data containing the neural activity.
        - is_recurrent: A boolean indicating whether the activity is recurrent or not. Always is. 
        - do_plot: A boolean indicating whether to plot the activity or not.
        
        Returns:
        - ep_activity: The neural activity data.
    '''
    if is_recurrent:
        ep_activity = pd.DataFrame(log['activity'])['rnn_hxs'].to_list()
    else:
        ep_activity = pd.DataFrame(log['activity'])['hx1_actor'].to_list()
    ep_activity = np.stack(ep_activity)

    if do_plot:
        ep_activity.shape
        fig, ax = plt.subplots(figsize=(15,5))
        ms = ax.matshow(ep_activity.T, cmap='RdBu')
        # plt.gca().invert_xaxis()
        plt.ylabel(f"Neurons")
        plt.xlabel(f"Time-step")
        plt.title(f"Neural activity over Time/Trajectory")

        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="1%", pad=0.05)
        plt.colorbar(ms, cax=cax)

    return ep_activity

def get_value(log):
    values = pd.DataFrame(log['activity'])['value'].to_list()
    values = np.array(values)
    return values


def pca_1episode(ep_activity, traj_df, twoD=True, threeD=False, colnames=None):
    if colnames is None:
        colnames = ['odor_obs', 
            'odor_01', 
            'odor_ma', 
            'odor_lastenc', 
            'wind_theta_obs', 
            'agent_angle_ground',
            'stray_distance', 
            'r_step', 
            ]

    pca = skld.PCA(3, whiten=False)
    pca.fit(ep_activity)
    X_pca = pca.transform(ep_activity)

    for colname in colnames:
      if twoD:
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        ax.plot(X_pca[:, 0], X_pca[:, 1],  linewidth=0.6, c='grey', alpha=0.5)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                s=20, c=traj_df[colname], cmap=plt.cm.get_cmap('RdBu'), vmin=0., vmax=1.)
        plt.title(f"State Space [{colname}]")
        ax.scatter(X_pca[0, 0], X_pca[0, 1], c='g', marker='o', s=100) # Start
        ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='g', marker='x', s=200) # End
        ax.set_xlabel(f'PC1 (VarExp: {pca.explained_variance_ratio_[0]:0.2f})')
        ax.set_ylabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
        ax.set_aspect('equal')
        
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)

        plt.tight_layout()
        plt.show()

      if threeD:
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca(projection='3d')
        ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], linewidth=0.6, c='grey', alpha=0.5)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                s=20, c=traj_df[colname], cmap=plt.cm.get_cmap('RdBu'), vmin=0, vmax=1)
        ax.scatter(X_pca[0, 0], X_pca[0, 1], X_pca[0, 2], c='g', marker='o', s=100) # Start
        ax.scatter(X_pca[-1, 0], X_pca[-1, 1], X_pca[-1, 2], c='g', marker='x', s=150) # End
        ax.set_xlabel(f'PC1 (VarExp: {pca.explained_variance_ratio_[0]:0.2f})')
        ax.set_ylabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
        ax.set_zlabel(f'PC3 (VarExp: {pca.explained_variance_ratio_[2]:0.2f})')
        plt.title(f"State Space [{colname}]")
        
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        # plt.colorbar(sc) # Didn't work!
        plt.tight_layout()
        plt.show()

def tsne_1episode(ep_activity, twoD=True, threeD=False, colnames=None):
    # TODO
    from sklearn.manifold import TSNE
    tsne = TSNE(2)
    X_tsne = tsne.fit_transform(ep_activity)

    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    ax.plot(X_tsne[:, 0], X_tsne[:, 1], linewidth=0.8)
    ax.scatter(X_tsne[0, 0], X_tsne[0, 1], c='r', marker='o', s=100) # Start
    ax.scatter(X_tsne[-1, 0], X_tsne[-1, 1], c='r', marker='x', s=150) # End
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.show()


def animate_activity_1episode(ep_activity, traj_df, episode_idx, 
    outprefix, fprefix, pca_dims=2, pca_common=None, invert_colors=False, title=True):
    assert pca_dims in [2, 3]
    # TODO implement 3D

    if pca_common is None:
        pca = skld.PCA(pca_dims, whiten=False)
        pca.fit(ep_activity)
    else:
        print("Using pca_common...")
        pca = pca_common

    X_pca = pca.transform(ep_activity)

    t_vals =  traj_df['t_val'] 
    if not os.path.exists(f'{outprefix}/tmp/'):
        os.makedirs(f'{outprefix}/tmp/')

    output_fnames = []
    for t_idx in tqdm.tqdm(range(X_pca.shape[0])):
        t_val = t_vals[t_idx]
        title_text = f"ep:{episode_idx} step:{t_idx} [t:{t_val:0.2f}]"

        if pca_dims == 2:
            fig = plt.figure(figsize=(6,6))
            ax = fig.gca()
            ax.plot(X_pca[:, 0], X_pca[:, 1],  linewidth=0.6, c='grey', alpha=0.5)
            sc = ax.scatter(X_pca[:t_idx, 0], X_pca[:t_idx, 1], s=15, 
                c=np.arange(1, t_idx+1)/(t_idx+1), cmap=plt.cm.get_cmap('Reds'), vmin=0., vmax=1.)
            ax.scatter(X_pca[0, 0], X_pca[0, 1], c='g', marker='o', s=100) # Start
            ax.scatter(X_pca[t_idx, 0], X_pca[t_idx, 1], c='g', marker='x', s=200) # End
            ax.set_aspect('equal')
        else: 
            fig = plt.figure(figsize=(12,6)) # 15,15 gets the whole plot by the front is too small then
            ax = fig.add_subplot(projection='3d')
            if invert_colors: # white dynamic trajectories and bigger moment to moment dots against the black background
                ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], linewidth=1, c='white', alpha=1)
                sc = ax.scatter(X_pca[:t_idx, 0], X_pca[:t_idx, 1], X_pca[:t_idx, 2], s=30, 
                    c=np.arange(1, t_idx+1)/(t_idx+1), cmap=plt.cm.get_cmap('Reds'), vmin=0., vmax=1.)
            else:
                ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], linewidth=0.6, c='grey', alpha=0.5)
                sc = ax.scatter(X_pca[:t_idx, 0], X_pca[:t_idx, 1], X_pca[:t_idx, 2], s=15, 
                    c=np.arange(1, t_idx+1)/(t_idx+1), cmap=plt.cm.get_cmap('Reds'), vmin=0., vmax=1.)

            ax.scatter(X_pca[0, 0], X_pca[0, 1], X_pca[0, 2], c='g', marker='o', s=100) # Start
            ax.scatter(X_pca[t_idx, 0], X_pca[t_idx, 1], X_pca[t_idx, 2], c='g', marker='x', s=150) # End
            ax.set_zlabel(f'PC3 (VarExp: {pca.explained_variance_ratio_[2]:0.2f})')

        ax.set_xlabel(f'PC1 (VarExp: {pca.explained_variance_ratio_[0]:0.2f})')
        ax.set_ylabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
        if title:
            plt.title(title_text)
        plt.tight_layout()
        if invert_colors:
            fig.set_facecolor('black')
            ax.set_facecolor('black')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            plt.rcParams['grid.color'] = "dimgrey"
            plt.rcParams['lines.linewidth'] = 0.1
            ax.tick_params(color='black', labelcolor='black')
            ax.grid(True)
            ax.set_xlabel(f'PC1')
            ax.set_ylabel(f'PC2')
            ax.set_zlabel(f'PC3')
        # # no ticks and truncated axis labels
        # ax.tick_params(color='white', labelcolor='white')
        # ax.set_xlabel(f'PC1')
        # ax.set_ylabel(f'PC2')
        # ax.set_zlabel(f'PC3')

        common_suffix = '_common' if pca_common is not None else '' 
        output_fname = f'{outprefix}/tmp/{fprefix}_pca{pca_dims}d{common_suffix}_ep{episode_idx}_step{t_idx:05d}.png'
        output_fnames.append(output_fname)
        # plt.savefig(output_fname, bbox_inches='tight')
        # plt.subplots_adjust(left=0.05, right=0.8, top=0.95, bottom=0.05)
        plt.savefig(output_fname, bbox_inches='tight', pad_inches=0.3)
        # release memory from matplotlib
        fig.clf()
        ax.cla()
        plt.close()
        # plt.savefig(output_fname)

    output_fnames = natsorted(output_fnames,reverse=False)
    clips = [ImageClip(f).set_duration(0.08) for f in output_fnames] # 
    concat_clip = concatenate_videoclips(clips, method="compose")
    fanim = f"{outprefix}/{fprefix}_pca{pca_dims}d{common_suffix}_ep{episode_idx:03d}.mp4"
    concat_clip.write_videofile(fanim, fps=15, verbose=False, logger=None)
    print("Saved", fanim)

    for f in output_fnames:
        # https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)


def get_activity_diff():
    # Neural activity Diff
    ep_activityT_reordered_diff = np.diff(ep_activityT_reordered)
    print(ep_activityT_reordered_diff.shape, ep_activityT_reordered.shape)
    ms = plt.matshow(ep_activityT_reordered_diff, cmap='RdBu')
    plt.ylabel(f"Neurons")
    plt.xlabel(f"Time-step")
    plt.title(f"Neural activity DIFF over Time/Trajectory")
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(ms, cax=cax)

    # TODO
    h_diff_norms = np.linalg.norm(ep_activityT_reordered_diff, axis=0)
    pd.Series(h_diff_norms).plot(figsize=(12,2), title=r"$\nabla h_t$");

def plot_activity_obs():
    # TODO
    # ms = plt.matshow(traj_df.T, cmap='RdBu')
    ms = plt.matshow(obs.T - obs.T.mean(axis=0), cmap='RdBu')
    plt.xlabel(f"Time-step")
    plt.title(f"trajectory DF")
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(ms, cax=cax)


#### Cluster and Disentangle the Dynamics ####
def get_actvity_reordering(ep_activity, do_plot=False):
    # fig = plt.figure(figsize=(1,1))
    clusts = sns.clustermap(ep_activity)
    reordering = clusts.dendrogram_col.reordered_ind
    ep_activityT_reordered = ep_activity.T[reordering,:]

    if do_plot:     
        ms = plt.matshow(ep_activityT_reordered, cmap='RdBu')
        plt.ylabel(f"Neurons")
        plt.xlabel(f"Time-step")
        plt.title(f"Neural activity over Time/Trajectory")
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="1%", pad=0.05)
        plt.colorbar(ms, cax=cax)

    return reordering, ep_activityT_reordered
