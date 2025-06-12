import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas
import scipy
import scipy as sp
import scipy.stats
from scipy.integrate import odeint
import multiprocessing
import time
import pickle
import multiprocessing
import numpy as np
import scipy.stats
import time
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
import tqdm
import config
np.random.seed(config.seed_global)


# from config import sim as simc
# from numba import jit # TODO


# wind/plume sim parameters
# dt = simc['dt']
# wind_magnitude = simc['wind_magnitude']
# wind_noisy = simc['wind_noisy'] # used by Inflexible/Floris generator
# birth_rate = simc['birth_rate']


def get_wind_vectors_flexible(T, wind_magnitude, local_state=None, regime=None):
    '''
    T: 1D array of timestamps
    local_state: need this to get unique noise arrays if running code in parallel
    '''
    if local_state is None:
        local_state = np.random.RandomState(0)

    # Setup baseline wind vectors: L --> R with 
    wind_degrees = np.zeros(len(T))
    wind_speeds = np.ones(len(T))*wind_magnitude #+ local_state.normal(0, 0.01*wind_magnitude, len(T))

    # 45 degree perturbation midway
    if 'switch' in regime:
        # TODO: Change this to happen at a certain fixed time
        # Since this is testing data, no need to randomize this time
        how_much = int(regime.replace('switch',''))
        wind_degrees_perturb = np.ones(len(T))*how_much
        wind_degrees_perturb[:int(len(T)/2)] = 0
        wind_degrees += wind_degrees_perturb

    # # Add random noise to wind speed
    # if 'noisy_speed' in features:
    #     wind_speeds += local_state.normal(0, 0.2*wind_magnitude, len(T))



    # Add random noise to wind degree
    if 'noisy' in regime:

        noise = np.zeros(len(T)) # Init
        repN = 100 # timesteps
        repN = 200 if 'noisy2' in regime else repN
        repN = 300 if 'noisy3' in regime else repN
        repN = 400 if 'noisy4' in regime else repN
        repN = 500 if 'noisy5' in regime else repN
        repN = 600 if 'noisy6' in regime else repN
        degz = 60 # +/- degz 

        # More evenly spaced
        switch_idxs = np.arange(len(T), step=repN, dtype=int)        
        switch_idxs = [ s + local_state.choice(np.arange(-int(repN/10), int(repN/10), dtype=int)) for s in switch_idxs ]
        switch_idxs = np.sort(switch_idxs)
        for idx in switch_idxs:
            noise[idx:] = local_state.normal(0, degz/2)


        # limit max
        noise = np.clip(noise, -degz, degz)        

        wind_degrees += noise # no smoothing

    # Add random noise to wind degree
    if 'poisson_noisy3' in regime:
        # note that poisson_mag_noisy3 does not get ran with this!!!! - 
        # based on /src/JH_boilerplate/check_noisy_wind_dirs.ipynb
        noise = np.zeros(len(T)) # Init
        degz = 60 # +/- degz 

        # Parameters used in sim_plume.sh
        lambda_ = 1 / 3  # Average rate (1 event every 3 seconds)
        # Time steps
        num_samples = len(T)
        local_state = np.random.RandomState(11)
        # Generate Poisson-distributed events
        events = local_state.poisson(lambda_ / 100, num_samples) # (average_hit_rate_per_sec / sampling_rate), len(time_series)
        # Convert to binary (0s and 1s)
        poisson_time_series = (events > 0).astype(int)
        # for logging purposes
        start_idx=60*100
        end_idx=103*100
        print(f"[LOG] avergae num wind changes per sec {sum(poisson_time_series[start_idx:end_idx]) / (end_idx-start_idx) * 100} from {start_idx/100} to {end_idx/100}")
        print(f"[LOG] # wind changes {sum(poisson_time_series[start_idx:end_idx])} from {start_idx/100} to {end_idx/100}")

        # which indices where the wind changes
        switch_idxs = np.where(poisson_time_series == 1)[0]

        for idx in switch_idxs:
            noise[idx:] = local_state.normal(0, degz/2)
            
        # limit max
        noise = np.clip(noise, -degz, degz)        

        wind_degrees += noise # no smoothing

    # Convert to X Y
    wind_x = np.cos( wind_degrees * np.pi / 180. )*wind_speeds
    wind_y = np.sin( wind_degrees * np.pi / 180. )*wind_speeds
    
    # 040425 added for varying wind magnitude 
    if 'mag' in regime:
        noise = np.zeros(len(T)) # Init

        # Parameters used in sim_plume.sh
        lambda_ = 1 / 3  # Average rate (1 event every 3 seconds)
        # Time steps
        num_samples = len(T)
        local_state = np.random.RandomState(11)
        # Generate Poisson-distributed events
        events = local_state.poisson(lambda_ / 100, num_samples) # (average_hit_rate_per_sec / sampling_rate), len(time_series)
        # Convert to binary (0s and 1s)
        poisson_time_series = (events > 0).astype(int)
        
        # Find indices where switches occur
        switch_idxs = np.where(poisson_time_series == 1)[0]

        # Wind magnitude bounds
        wind_mag_max = 0.7 # mag_narrow = 0.7; mag 1.5
        wind_mag_min = 0.3 # mag_narrow = 0.3; mag 0.1
        max_factor = wind_mag_max / wind_magnitude
        min_factor = wind_mag_min / wind_magnitude
        
        # Sample N wind magnitudes, where N is the number of switch points
        N = len(switch_idxs)
        sampled_magnitude_ratios = np.random.uniform(min_factor, max_factor, N)
    

        # Fill in the wind_magnitude array by repeating each sampled magnitude
        # until the next switch point
        wind_speed_ratios = np.ones(len(T))
        for i in range(N):
            # Current switch index
            start_idx = switch_idxs[i]
            
            # End index is either the next switch point or the end of the array
            end_idx = switch_idxs[i+1] if i < N-1 else len(T)
            # Fill this segment with the sampled magnitude
            wind_speed_ratios[start_idx:end_idx] = sampled_magnitude_ratios[i]
            # print(f'factor = {sampled_magnitude_ratios[i]}, scaled mag = {wind_magnitude[start_idx:start_idx+1]}') 
        # Now apply the wind magnitude to wind components
        wind_x = wind_x * wind_speed_ratios
        wind_y = wind_y * wind_speed_ratios

    return wind_x, wind_y


def get_wind_xyt(duration, dt, wind_magnitude, verbose=True, regime='noisy3'):
    T = np.arange(0, duration, dt).astype('float64')
    if verbose:
        print("Generate and save wind data ... ")

    # if not flexible:
    #     wind_x, wind_y = get_wind_vectors_original(T, 
    #         wind_magnitude=wind_magnitude, 
    #         noisy=wind_noisy, 
    #         switch_direction=switch_direction)

    if 'eval' in regime:
        local_state = np.random.RandomState(6) # seed for eval_noisy with a good range of wind directions 
    else:
        local_state = None # init in function 
    wind_x, wind_y = get_wind_vectors_flexible(T, wind_magnitude, regime=regime, local_state=local_state)

    data_wind = pd.DataFrame({'wind_x': wind_x[0:len(T)],
                              'wind_y': wind_y[0:len(T)],
                              'time': T})

    # Compress numerical data!
    n_times_pre = len(data_wind['time'].unique())
    for col in data_wind.select_dtypes(include='float').columns:
        if 'time' in col:
            data_wind[col] = data_wind[col].astype('float64') 
        else:    
            data_wind[col] = data_wind[col].astype('float16') # Use lightest floats
    assert n_times_pre == len(data_wind['time'].unique()) # Make sure quantization is lossless

    return data_wind


# Puff Simulation ODEINT
def puff_wind_diff_eq(xyr, t, *args):
    dt, wind_x, wind_y = args
    x,y,r = xyr

    idx = int(t/dt)
    if idx > len(wind_x)-1:
        idx = -1
    xdot = wind_x[idx]
    ydot = wind_y[idx]

    rdot = 0.01

    return [xdot, ydot, rdot]   

# Serial (not parallel) version
def integrate_puff_from_birth(args):
    T, wind_x, wind_y, birth_index, seed = args
    # Simulate once

    xyr_0 = [0,0,0.01] # initial x, y, radius
    dt = 0.01
    # wind_x, wind_y = get_wind_vectors_original(T, 
    #     local_state=local_state, 
    #     wind_magnitude=wind_magnitude, 
    #     switch_direction=switch_direction)

    # # Add some y-direction variation per puff
    # local_state = np.random.RandomState(seed)
    # wind_y_var = local_state.normal(0, wind_magnitude, len(T))

    vals, extra = odeint(puff_wind_diff_eq, xyr_0, T[birth_index:], 
                         args=(dt, wind_x, wind_y), 
                         full_output=True)
    Z = np.zeros([birth_index, 3])
    return np.vstack((Z, vals))

def get_puffs_raw(T, wind_x, wind_y, birth_rate, ncores=2, verbose=True):
    #### 
    print("Generating indices of plume puff births...")
    births = scipy.stats.poisson.rvs(birth_rate, size=len(T))
    birth_indices = []
    for idx, birth in enumerate(births):
        birth_indices.extend([idx]*birth)

    #### 
    print("Starting parallel simulation ({} cores) for each plume puff...".format(ncores))
    # Setup inputs for parallel simulation
    seeds = np.arange(0, len(birth_indices))
    inputs = [[T,
        wind_x,
        wind_y, 
        birth_indices[i], 
        seeds[i]] for i in range(0, len(birth_indices))]

    t_start = time.time()

    pool = multiprocessing.Pool(ncores)
    puffs = pool.map(integrate_puff_from_birth, inputs)
    comp_time = time.time() - t_start
    if verbose:
        print('Computation time: ', comp_time)
        print('ncores: ', ncores, ' puffs: ', len(puffs), ' steps: ', len(T))
        print('Computation time (C) per core (n) per puff (p) per step (s) = (C*n/p*s) = ', ncores*comp_time/(len(puffs)*len(T)))
        print('Total time = C*n/p*s')

    puffs_arr = np.stack(puffs)

    if save:
        if verbose:
            print("Save raw puff data...")
        data = {'puffs': puffs_arr, 'wind_x': wind_x, 'wind_y': wind_y, 'time': T}
        with open('puff_data_array.pickle', 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    return data

##### Parallel DF version #####
def integrate_puff_from_birth_df(args):
    T, tidxs, wind_x, wind_y, wind_y_var, birth_index, puff_index, seed = args

    xyr_0 = [0,0,0.01] # initial x, y, radius
    dt = 0.01

    # Add some y-direction variation per puff
    local_state = np.random.RandomState(seed)
    wind_dy = local_state.normal(0, wind_y_var, len(wind_y))


    vals, extra = odeint(puff_wind_diff_eq, xyr_0, T[birth_index:], 
                         args=(dt, wind_x, wind_y + wind_dy), 
                         full_output=True)

    puff_df = pd.DataFrame({
        'puff_number': puff_index, 
        'time': T[birth_index:], 
        'tidx': tidxs[birth_index:], 
        'x': vals[:, 0], 
        'y': vals[:, 1], 
        'radius': vals[:, 2],
        })

    # Postprocessing 
    puff_df = puff_df.query("(radius != 0) & (x<10) & (y<10) & (x>-2) & (y>-10)")

    return puff_df

def get_puffs_df_oneshot(wind_df, wind_y_var, birth_rate, ncores=2, verbose=True):
    T = wind_df['time'].to_numpy()
    tidxs = wind_df['tidx'].to_numpy()
    wind_x = wind_df['wind_x'].to_numpy()
    wind_y = wind_df['wind_y'].to_numpy()

    #### 
    print("Generating indices of plume puff births...")
    births = scipy.stats.poisson.rvs(birth_rate, size=len(T))
    birth_indices = []
    for idx, birth in enumerate(births):
        birth_indices.extend([idx]*birth)

    #### 
    print("Starting parallel simulation ({} cores) for each of {} plume puffs...".format(ncores, len(birth_indices)))
    # Setup inputs for parallel simulation
    seeds = np.arange(0, len(birth_indices))
    puff_indices = np.arange(0, len(birth_indices))
    # OVERRIDE wind_y_var
    # wind_y_var = np.linalg.norm([wind_x, wind_y])/wind_y_varx
    inputs = [[T,
        tidxs,
        wind_x,
        wind_y, 
        wind_y_var,
        birth_indices[i], 
        puff_indices[i], # could just use i
        seeds[i], # could just use i
        ] for i in range(0, len(birth_indices))]

    t_start = time.time()
    pool = multiprocessing.Pool(ncores)
    # puff_dfs = pool.map(integrate_puff_from_birth_df, inputs)
    puff_dfs = list(tqdm.tqdm(pool.imap(integrate_puff_from_birth_df, inputs)))
    comp_time = time.time() - t_start
    if verbose:
        print('Computation time: ', comp_time)
        print('ncores: ', ncores, ' puffs: ', len(birth_indices), ' steps: ', len(T))
        print('Computation time (C) per core (n) per puff (p) per step (s) = (C*n/p*s) = ', ncores*comp_time/(len(birth_indices)*len(T)))
        print('Total time = C*n/p*s')

    t_start = time.time()
    puffs_df = pd.concat(puff_dfs)
    comp_time = time.time() - t_start
    print('Time to concatenate {} puff dataframes: {}'.format(len(puff_dfs), comp_time))

    n_times_pre = len(puffs_df['time'].unique())
    for col in puffs_df.select_dtypes(include='float').columns:
        if 'time' in col:
            puffs_df[col] = puffs_df[col].astype('float64') # time needs a heavier float
        else:
            puffs_df[col] = puffs_df[col].astype('float16') # Use lightest floats
    assert n_times_pre == len(puffs_df['time'].unique()) # Make sure compression lossless

    return puffs_df


#### Faster vectorized version ####
def gen_puff_dict(puff_number, tidx):
    return {
     'puff_number': puff_number,
     'time': np.float64(tidx)/100.,
     'x': 0.0,
     'y': 0.0,
     'radius': 0.01,
     # 'x_minus_radius': -0.01,
     # 'x_plus_radius': 0.01,
     # 'y_minus_radius': -0.01,
     # 'y_plus_radius': 0.01,
     # 'concentration': 1.0,
     'tidx': tidx,
    }
    
def grow_puffs(birth_rate, puff_t, tidx):
    num_births = sp.stats.poisson.rvs(birth_rate, size=1)[0]
    puff_number = puff_t['puff_number'].max() + 1
    
    new_rows = [ gen_puff_dict(puff_number+i, tidx) for i in range(num_births)]    
    new_rows = pd.DataFrame( new_rows )
    return pd.concat([puff_t, new_rows])
    
def manual_integrator(puff_t, wind_t, tidx,
                      dt=np.float64(0.01), 
                      rdot=0.01, 
                      birth_rate=1.0, 
                      min_radius=0.01, 
                      wind_y_var=0.5):
    n_puffs = len(puff_t)
    puff_t['x'] += wind_t['wind_x'].item()*dt
    puff_t['y'] += wind_t['wind_y'].item()*dt + np.random.normal(0, wind_y_var, size=n_puffs)*dt
    puff_t['radius'] += dt * rdot
    puff_t['tidx'] = tidx
    puff_t['time'] = wind_t['time'].item()
    
    # Trim plume
    puff_t = puff_t.query("(radius > 0) & (x<10) & (y<10) & (x>-2) & (y>-10)")
    # max_extent = 10
    # puff_t = puff_t.query(f"(radius > 0) & (abs(x) < {max_extent}) & (abs(y) < {max_extent})")

    # Grow plume
    puff_t = grow_puffs(birth_rate, puff_t, tidx)
    
    return puff_t

def get_puffs_df_vector(wind_df, wind_y_var, birth_rate, verbose=True):
    """Fast vectorized euler stepper"""
    print(wind_df.shape, wind_y_var, birth_rate)
    # Initialize
    n_steps = int((wind_df['time'].max() - wind_df['time'].min())*100)
    tidx = 0
    puff_t = pd.DataFrame([gen_puff_dict(puff_number=0, tidx=tidx)])
    wind_t = wind_df.query("tidx == @tidx").copy(deep=True).reset_index(drop=True)

    # Main Euler integrator loop
    puff_dfs = []
    for i in tqdm.tqdm(range(n_steps)):
        puff_t = manual_integrator(puff_t, wind_t, tidx, 
            birth_rate=birth_rate, wind_y_var=wind_y_var)
        puff_dfs.append( puff_t )

        tidx += 1
        wind_t = wind_df.query("tidx == @tidx").copy(deep=True).reset_index(drop=True)
        if wind_t.shape[0] != 1:
            print("Likely numerical error!:", tidx, wind_t)

    # Gather data and post-process float format
    t_start = time.time()
    puffs_df = pd.concat(puff_dfs)
    if verbose:
        comp_time = time.time() - t_start
        print('Time to concatenate {} puff dataframes: {}'.format(len(puff_dfs), comp_time))

    n_times_pre = len(puffs_df['time'].unique())
    for col in puffs_df.select_dtypes(include='float').columns:
        if 'time' in col:
            puffs_df[col] = puffs_df[col].astype('float64') # time needs a heavier float
        else:
            puffs_df[col] = puffs_df[col].astype('float16') # Use lightest floats
    assert n_times_pre == len(puffs_df['time'].unique()) # Make sure compression lossless
    return puffs_df


# from plume.sim_analysis for plot_puffs_and_wind_vectors

# New version: One circle at hardcoded (x,y) with quiver 
def plot_wind_vectors(data_puffs, data_wind, t_val, ax, invert_colors=False):
    # Instantaneous wind velocity
    # Normalize wind (just care about angle)
    data_at_t = data_wind[data_wind.time==t_val]
    v_x, v_y = data_at_t.wind_x.mean(), data_at_t.wind_y.mean()
    v_xy = np.sqrt(v_x**2 + v_y**2)*20
    v_x, v_y = v_x/v_xy, v_y/v_xy
    # print("v_x, v_y", v_x, v_y)

    # Arrow
    x,y = -0.15, 0.6 # Arrow Center [Note usu. xlim=(-0.5, 8)]
    if invert_colors:
        color='white'
    else:
        color='black'
    ax.quiver(x, y, v_x, v_y, color=color, scale=2.5)
    # ax.quiver(x, y, v_x, v_y, color='black', scale=500)

    # Circle is 1 scatterplot point!
    ax.scatter(x, y, s=500, 
        facecolors='none', 
        edgecolors=color,
        linestyle='--')

def plot_puffs(data, t_val, ax=None, show=True):
    # TODO check color to concentration mapping
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
        
    # xmin = -2 #data.x.min()
    # xmax = 12 #data.x.max()
    # ymin = -5 #data.y.min()
    # ymax = +5 #data.y.max()
    # set limits
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

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
    k = 6250*((fig.get_figwidth()/8.0)**2) # trial-and-error
    s = k*(data_at_t.radius)**2 
    # print('size', s) # 885

    ax.scatter(data_at_t.x, data_at_t.y, s=s, facecolor=rgba_colors, edgecolor='none')

    if show:
        plt.show()

def plot_puffs_and_wind_vectors(data_puffs, data_wind, t_val, ax=None, fname='', plotsize=(10,10), show=True, invert_colors=False):
    if ax is None:
        fig = plt.figure(figsize=plotsize)
        ax = fig.add_subplot(111)
    plot_wind_vectors(data_puffs, data_wind, t_val, ax, invert_colors=invert_colors)
    plot_puffs(data_puffs, t_val, ax, show=False)
    
    if len(fname) > 0:
        # fname = savedir + '/' + 'puff_animation_' + str(idx).zfill(int(np.log10(data['puffs'].shape[1]))+1) + '.jpg'
        fig.savefig(fname, format='jpg', bbox_inches='tight')
        plt.close()
    return fig, ax