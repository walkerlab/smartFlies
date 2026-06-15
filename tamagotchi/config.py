import os
import sys
import glob
import numpy as np
import matplotlib

import socket
MACHINE = socket.gethostname().lower()
curr_wd = os.getcwd()
# datadir = '/src/data/performance_plot_data_code/collection_1/'
# datadir = '/src/data/performance_plot_data_code/lawrence/' # for running Toha comparison experiments
if 'gscratch' in curr_wd and 'walkerlab' in curr_wd:
    datadir = '/gscratch/walkerlab/jqhu/smartFlies/data/published_results/reproduce/'
elif 'gscratch' in curr_wd and 'portia' in curr_wd:
	datadir = '/gscratch/portia/jqhu/work/active_sensing/smartFlies/data/published_results/reproduce/'
elif '/src' in curr_wd:
    datadir = '/src/data/published_results/reproduce/'
else:
	raise ValueError(f'Unrecognized gscratch path: {curr_wd}')

# print(f'Using datadir: {datadir}', flush=True)
# if MACHINE == 'mycroft':
# 	datadir = '/data/users/satsingh/plumedata/'
# if (MACHINE == 'salarian') or (MACHINE == 'cylon'):
# 	datadir = '/data1/users/satsingh/plumedata/'

seed_global = 137

traj_colormap = { 
	# 'on': 'lime',
	# 'on': 'darkgreen',
	# 'on': 'seagreen', # manuscript
	'on': 'gold', # Bing presentation
	# 'on': 'mediumseagreen',
	# 'on': 'royalblue',
	# 'on': 'dodgerblue',
	# 'on': 'blue',

	# 'off': 'blue', # manuscript	
	# 'off': 'mediumorchid', # Bing presentation
	'off': 'hotpink', # Bing presentation
	# 'off': 'dodgerblue', # lighter than royalblue
	# 'off': 'royalblue',
	# 'off': 'brown',
	# 'off': 'crimson',
	# 'off': 'red',
}

regime_colormap = {
					'SEARCH': 'red', 
					# 'SEARCH': 'brown', 

                   # 'TRACK':'darkolivegreen', # darker
                   # 'TRACK':'forestgreen', # standard green
                   'TRACK':'seagreen', # just right
                   # 'TRACK':'limegreen', 
                   
                   # 'RECOVER':'mediumslateblue', 	
                   'RECOVER':'slateblue', 
                   }
outcome_colormap = {'HOME': 'g', 
				    'OOB':'r', 
				    'OOT':'b'}

ttcs_colormap = {'HOME': 'b', 'OOB':'darkorange'}


plume_color = matplotlib.colors.to_rgba('gray')
# from sim_analysis.py
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('gray')[:3] # decent
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkgray')[:3] # decent
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('dimgray')[:3] # decent
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkslategray')[:3] # too dark
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightsteelblue')[:3] # ok
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('red')[:3] 
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightskyblue')[:3] 



# mwidth, mheight = 5.5, 9 # Manuscript usable dimensions for NeurIPS/ICLR
mwidth, mheight = 7, 9 # Manuscript usable dimensions for IEEE

# metadata associated with some seeds
seedmeta = {
	'2760377': {'recover_min':12, 'recover_max': 30, },
	# '3199993': {'recover_min':12, 'recover_max': 25, },
	'3307e9': {'recover_min':12, 'recover_max': 35, },
	'541058': {'recover_min':12, 'recover_max': 38, },
	# '9781ba': {'recover_min':12, 'recover_max': 25, },
}


env = {
	# 'rescale': False,
	# 'sim_steps_max': 300, 
	# 'reset_offset_tmax': 60.0 - 300.0/25, # t_val_min - sim_steps_max/fps
	# 'reset_offset_tmax': 25.00, # seconds
	# 'homed_radius': 0.2, # meters
	# 'stray_distance': 2.0, # meters
    'odor_threshold': 0.0001, # arbit units
    # 'odor_threshold': 1e-8, # arbit units
	'arena_bounds': {
		'x_min':-5, 
		'x_max':20, 
      	'y_min':-5, 
      	'y_max':5
      	},	

	# Max agent CW/CCW turn per second
	# 'turn_capacity': 25*np.pi * 0.75, 
	
	# Max agent speed in m/s
	# 'move_capacity': 2.5, 	
	# 'curriculum': True, # set in cli train
	# 'difficulty': 0.5, # Curriculum difficulty \in [0.0, 1.0]
	# 'difficulty': 0.65, # Curriculum difficulty \in [0.0, 1.0]
}

# Coefficients for PlumeEnvironment_v3 action_physics=='force'.
# The agent commands thrust/torque (normalized to [-1, 1]) and the env integrates
# rigid-body dynamics with substeps. SI units; defaults give a ~1 mg fly with
# boosted thrust reaching ~2 m/s in still air. control_rate is taken from the
# env's own dt (1/env_dt), and wind is supplied by the plume data (not modeled here).
force_physics = {
	# control limits (thrust forces / yaw torque)
	'T_para_max':   6.0e-5,    # N, parallel (forward/back, body frame)
	'T_perp_max':   2.0e-5,    # N, perpendicular (lateral, body frame)
	'tau_max':      3.0e-10,   # N*m, yaw torque (ccw positive)
	# translation
	'mass':         1.0e-6,    # kg
	'drag':         3.0e-5,    # N*s/m, linear drag acting on airspeed
	# rotation
	'inertia':      5.0e-13,   # kg*m^2
	'k_rot':        2.0e-11,   # N*m*s, rotational drag
	# numerics
	'physics_substeps': 4,     # zero-order-hold substeps per env step
}

# for data_utils.plot_artifacts
mlflow_colors = {
            'constantx5b5': 'blue',
            'constant_mag_narrowx5b5': 'green',
            'constant_jitterx5b5': 'red',
			# 'constant_magx5b5': 'red',
            'noisy3x5b5': 'purple',
            'noisy_jitterx5b5': 'purple',
            'poisson_mag_narrow_noisy3x5b5': 'orange',
            'poisson_mag_noisy3x5b5': 'cyan',
            'poisson_noisy3x5b5': 'magenta'
        }