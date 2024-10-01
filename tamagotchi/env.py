##### from plume/plume_env.py  ##### from plume/plume_env.py  ##### from plume/plume_env.py
from data_util import load_plume, get_concentration_at_tidx
import config as config
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import pandas as pd
import numpy as np
import torch
import gym
from gym import spaces
from pprint import pprint
from scipy.spatial.distance import cdist 
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv as SubprocVecEnv_
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_
import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
import copy

class PlumeEnvironment(gym.Env):
  """
  Documentation: https://gym.openai.com/docs/#environments
  Plume tracking
  """
  def __init__(self, 
    t_val_min=60.00, 
    sim_steps_max=300, # steps
    reset_offset_tmax=30, # seconds; max secs for initial offset from t_val_min
    dataset='constantx5b5',
    move_capacity=2.0, # Max agent speed in m/s
    turn_capacity=6.25*np.pi, # Max agent CW/CCW turn per second
    wind_obsx=1.0, # normalize/divide wind observations by this quantity (move_capacity + wind_max) 
    movex=1.0, # move_max multiplier for tuning
    turnx=1.0, # turn_max multiplier for tuning
    birthx=1.0, # per-episode puff birth rate sparsity minimum
    birthx_max=1.0, # overall odor puff birth rate sparsity max
    env_dt=0.04,
    loc_algo='quantile',
    qvar=1.0, # Variance of init. location; higher = more off-plume initializations
    time_algo='uniform',
    angle_algo='uniform',
    homed_radius=0.2, # meters, at which to end flying episode
    stray_max=2.0, # meters, max distance agent can stray from plume
    wind_rel=True, # Agent senses relative wind speed (not ground speed)
    auto_movex=False, # simple autocurricula for movex
    auto_reward=False, # simple autocurricula for reward decay
    diff_max=0.8, # teacher curriculum; sets the quantile of init x location 
    diff_min=0.4, # teacher curriculum; sets the quantile of init x location 
    r_shaping=['step', 'oob'], # 'step', 'end'
    rewardx=1.0, # scale reward for e.g. A3C
    rescale=False, # rescale/normalize input/outputs [redundant?]
    squash_action=False, # apply tanh and rescale (useful with PPO)
    walking=False,
    walk_move=0.05, # m/s (x100 for cm/s)
    walk_turn=1.0*np.pi, # radians/sec
    radiusx=1.0, 
    diffusion_min=1.00, 
    diffusion_max=1.00, 
    action_feedback=False,
    flipping=False, # Generalization/reduce training data bias
    odor_scaling=False, # Generalization/reduce training data bias
    obs_noise=0.0, # Multiplicative: Wind & Odor observation noise.
    act_noise=0.0, # Multiplicative: Move & Turn action noise.
    dynamic=False,
    seed=137,
    verbose=0):
    super(PlumeEnvironment, self).__init__()

    assert dynamic is False
    np.random.seed(seed)    
    self.arguments = locals()
    print("PlumeEnvironment:", self.arguments)
    
    self.verbose = verbose
    self.venv = self
    self.walking = walking
    self.rewardx = rewardx
    self.rescale = rescale
    self.odor_scaling = odor_scaling
    self.stray_max = stray_max
    self.wind_obsx = wind_obsx
    self.reset_offset_tmax = reset_offset_tmax
    self.action_feedback = action_feedback
    self.qvar = qvar
    self.squash_action = squash_action
    self.obs_noise = obs_noise
    self.act_noise = act_noise
    if self.squash_action:
        print("Squashing actions to 0-1")

    # Fixed evaluation related:
    self.fixed_time_offset = 0.0 # seconds
    self.fixed_angle = 0.0 # downwind
    self.fixed_x = 7.0 
    self.fixed_y = 0.0 # might not work for switch/noisy! 


    # Environment/state variables
    # self.dt = config.env['dt'] 
    self.dt = env_dt # 0.1, 0.2, 0.4, 0.5 sec
    # self.fps = config.env['fps'] # 20/25/50/100 steps/sec
    self.fps = int(1/self.dt)
    # self.sim_fps = 100 # not used
    self.episode_step = 0 # skip_steps done during loading

    # Load simulated data
    self.radiusx = radiusx
    self.birthx = birthx
    self.birthx_max = birthx_max
    self.diffusion_max = diffusion_max # Puff diffusion multiplier (initial)
    self.diffusion_min = diffusion_min # Puff diffusion multiplier (reset-time)
    self.t_val_min = t_val_min
    self.episode_steps_max = sim_steps_max # Short training episodes to gather rewards
    self.t_val_max = self.t_val_min + self.reset_offset_tmax + 1.0*self.episode_steps_max/self.fps + 1.00

    self.set_dataset(dataset)

    # Correction for short simulations
    if self.data_wind.shape[0] < self.episode_steps_max:
      if self.verbose > 0:
        print("Wind data available only up to {} steps".format(self.data_wind.shape[0]))
      self.episode_steps_max = self.data_wind.shape[0]

    # Other initializations -- many redundant, see .reset() 
    # self.agent_location = np.array([1, 0]) # TODO: Smarter
    self.agent_location = None
    self.agent_location_last = self.agent_location
    self.agent_location_init = self.agent_location
    random_angle = np.pi * np.random.uniform(0, 2)
    self.agent_angle_radians = [np.cos(random_angle), np.sin(random_angle)] # Sin and Cos of angle of orientation
    self.step_offset = 0 # random offset per trial in reset()
    self.t_val = self.t_vals[self.episode_step + self.step_offset] 
    self.tidx = self.tidxs[self.episode_step + self.step_offset] 
    self.tidx_min_episode = self.tidx
    self.tidx_max_episode = self.tidx
    self.wind_ground = None
    self.stray_distance = 0
    self.stray_distance_last = 0
    self.agent_velocity_last = np.array([0, 0]) # Maintain last timestep velocity (in absolute coordinates) for relative sensory observations
    self.episode_reward = 0

    # Generalization & curricula
    self.r_shaping = r_shaping
    print("Reward Shaping", self.r_shaping)
    self.flipping = flipping 
    self.flipx = 1.0 # flip puffs around x-axis? 
    self.difficulty = diff_max # Curriculum
    self.diff_max = diff_max # Curriculum
    self.diff_min = diff_min # Curriculum
    self.odorx = 1.0 # Doesn't make a difference except when thresholding
    self.turnx = turnx
    self.movex = movex
    self.auto_movex = auto_movex
    self.auto_reward = auto_reward
    self.reward_decay = 1.00
    self.loc_algo = loc_algo
    self.angle_algo = angle_algo
    self.time_algo = time_algo
    assert self.time_algo in ['uniform', 'linear', 'fixed']
    self.outcomes = [] # store outcome last N episodes

    # Constants
    self.wind_rel = wind_rel
    self.turn_capacity = turn_capacity
    self.move_capacity = move_capacity 
    # self.turn_capacity = 1.0 * np.pi # Max agent can turn CW/CCW in one timestep
    # self.move_capacity = 0.025 # Max agent can move in one timestep
    self.arena_bounds = config.env['arena_bounds'] 
    self.homed_radius = homed_radius  # End session if dist(agent - source) < homed_radius
    self.rewards = {
      'tick': -10/self.episode_steps_max,
      'homed': 101.0,
      }


    # Define action and observation spaces
    # Actions:
    # Move [0, 1], with 0.0 = no movement
    # Turn [0, 1], with 0.5 = no turn... maybe change to [-1, 1]
    self.action_space = spaces.Box(low=0, high=+1,
                                        shape=(2,), dtype=np.float32)
    if self.rescale:
        ## Rescaled to [-1,+1] to follow best-practices: 
        # https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
        # Both will first clip to [-1,+1] then map to [0,1] with all other code remaining same
        self.action_space = spaces.Box(low=-1, high=+1,
                                        shape=(2,), dtype=np.float32)

    # Observations
    # Wind velocity [-1, 1] * 2, Odor concentration [0, 1]
    obs_dim = 3 if not self.action_feedback else 3+2
    self.observation_space = spaces.Box(low=-1, high=+1,
                                        shape=(obs_dim,), dtype=np.float32)

    ######## Experimental "walking mode" ########
    if self.walking:
        self.turn_capacity = walk_turn 
        self.move_capacity = walk_move 
        self.homed_radius = 0.02 # m i.e. 18cm walk from 0.20m (flying "homed" distance)
        self.stray_max = 0.05 # meters
        # self.rewards['tick'] = -1/self.episode_steps_max

  def update_env_param(self, params):
      for k,v in params.items():
          setattr(self, k, v)

  def set_dataset(self, dataset):
    self.dataset = dataset
    self.data_puffs_all, self.data_wind_all = load_plume(
        dataset=self.dataset, 
        t_val_min=self.t_val_min, 
        t_val_max=self.t_val_max,
        env_dt=self.dt,
        puff_sparsity=np.clip(self.birthx_max, a_min=0.01, a_max=1.00),
        diffusion_multiplier=self.diffusion_max,
        radius_multiplier=self.radiusx,
        )
    if self.walking:
        self.data_puffs_all = self.data_puffs_all.query('x <= 0.5')
    self.data_puffs = self.data_puffs_all.copy() # trim this per episode
    self.data_wind = self.data_wind_all.copy() # trim/flip this per episode
    self.t_vals = self.data_wind['time'].tolist()
    print("wind: t_val_diff", (self.t_vals[2] - self.t_vals[1]), "env_dt", self.dt)
    t_vals_puffs = self.data_puffs['time'].unique()
    print("puffs: t_val_diff", (t_vals_puffs[2] - t_vals_puffs[1]), "env_dt", self.dt)
    self.tidxs = self.data_wind['tidx'].tolist()

  def reload_dataset(self):
    self.set_dataset(self.dataset)

  def set_difficulty(self, level, verbose=True): # Curriculum
    """
    Location distance as a form of curriculum learning
    :level: in [0.0, 1.0] with 0.0 being easiest
    """
    if level < 0:
        self.difficulty = self.diff_max
    else:
        level = np.clip(level, 0.0, 1.0)
        self.difficulty = level
    if verbose:
        print("set_difficulty to", self.difficulty)

  def sense_environment(self):
    if (self.verbose > 1) and (self.episode_step >= self.episode_steps_max): # Debug mode
        pprint(vars(self))

    # Wind
    wind_absolute = self.wind_ground # updated by step()
    
    # Subtract agent velocity to convert to (observed) relative velocity
    if self.wind_rel: 
        wind_absolute = self.wind_ground - self.agent_velocity_last # TODO: variable should be named wind_relative

    # Get wind relative angle
    agent_angle_radians = np.angle( self.agent_angle[0] + 1j*self.agent_angle[1], deg=False )
    wind_angle_radians = np.angle( wind_absolute[0] + 1j*wind_absolute[1], deg=False )
    wind_relative_angle_radians = wind_angle_radians - agent_angle_radians
    wind_observation = [ np.cos(wind_relative_angle_radians), np.sin(wind_relative_angle_radians) ]    
    # Un-normalize wind observation by multiplying by magnitude
    wind_magnitude = np.linalg.norm(np.array( wind_absolute ))/self.wind_obsx
    wind_observation = [ x*wind_magnitude for x in wind_observation ] # convert back to velocity
    # Add observation noise
    wind_observation = [ x*(1.0+np.random.uniform(-self.obs_noise, +self.obs_noise)) for x in wind_observation ]

    if self.verbose > 1:
        print('wind_observation', wind_observation)
        print('t_val', self.t_val)


    odor_observation = get_concentration_at_tidx(
        self.data_puffs, self.tidx, self.agent_location[0], self.agent_location[1])
    if self.verbose > 1:
        print('odor_observation', odor_observation)
    if self.odor_scaling:
        odor_observation *= self.odorx # Random scaling to improve generalization 
    odor_observation *= 1.0 + np.random.uniform(-self.obs_noise, +self.obs_noise) # Add observation noise

    odor_observation = 0.0 if odor_observation < config.env['odor_threshold'] else odor_observation
    odor_observation = np.clip(odor_observation, 0.0, 1.0) # clip

    # Return
    observation = np.array(wind_observation + [odor_observation]).astype(np.float32) # per Gym spec
    if self.verbose > 1:
        print('observation', observation)
    return observation

  def get_abunchofpuffs(self, max_samples=300):  
    # Z = self.data_puffs[self.data_puffs.time==self.t_val].loc[:,['x','y']]
    # Z = self.data_puffs[self.data_puffs.tidx==self.tidx].loc[:,['x','y']]
    Z = self.data_puffs.query(f"tidx == {self.tidx}").loc[:,['x','y']]
    Z = Z.sample(n=max_samples, replace=False) if Z.shape[0] > max_samples else Z
    return Z

  def get_stray_distance(self):
    Z = self.get_abunchofpuffs()
    Y = cdist(Z.to_numpy(), np.expand_dims(self.agent_location,axis=0), metric='euclidean')
    try:
        minY = min(Y) 
    except Exception as ex:
        print(f"[PE_v1 get_stray_distance] Exception: {ex}, t:{self.t_val:.2f}, tidx:{self.tidx}({self.tidx_min_episode}...{self.tidx_max_episode}), ep_step:{self.episode_step}, {Z}")  
        minY = np.array([0])      
    return minY[0] # return float not float-array

  def get_initial_location(self, algo):
    loc_xy = None
    if 'uniform' in algo:
        loc_xy = np.array([
            2 + np.random.uniform(-1, 1), 
            np.random.uniform(-0.5, 0.5)])

        if self.walking:
            loc_xy = np.array([
              0.2 + np.random.uniform(-0.1, 0.1), 
              np.random.uniform(-0.05, 0.05)])

    if 'linear' in algo:
        # TODO
        loc_xy = np.array([
            2 + np.random.uniform(-1, 1), 
            np.random.uniform(-0.5, 0.5)])

    if 'quantile' in algo:
        """ 
        Distance curriculum
        Start the agent at a location with random location with mean and var
        decided by distribution/percentile of puffs 
        """
        q_curriculum = np.random.uniform(self.diff_min, self.diff_max)

        Z = self.get_abunchofpuffs()
        X_pcts = Z['x'].quantile([q_curriculum-0.1, q_curriculum]).to_numpy()
        X_mean, X_var = X_pcts[1], X_pcts[1] - X_pcts[0]
        # print("initial X mean, var, q: ", X_mean, X_var, q_curriculum)
        Y_pcts = Z.query("(x >= (@X_mean - @X_var)) and (x <= (@X_mean + @X_var))")['y'].quantile([0.05,0.5]).to_numpy()
        Y_pcts
        Y_mean, Y_var = Y_pcts[1], min(1, Y_pcts[1] - Y_pcts[0]) # TODO: What was min for?
        # print(Y_mean, Y_var)
        varx = self.qvar 
        # if 'switch' in self.dataset: # Preferably start within/close to plume
        #     varx = 0.1
        loc_xy = np.array([X_mean + varx*X_var*np.random.randn(), 
            Y_mean + varx*Y_var*np.random.randn()]) 

    if 'fixed' in algo:
        loc_xy = np.array( [self.fixed_x, self.fixed_y] )

    return loc_xy

  def get_initial_step_offset(self, algo):
    """ Time curriculum """
    if 'uniform' in algo:
        step_offset = int(self.fps * np.random.uniform(low=0.00, high=self.reset_offset_tmax))

    if 'linear' in algo:
        window = 5 # seconds
        mean = window + self.difficulty*(self.reset_offset_tmax-window)
        step_offset = int(self.fps * np.random.uniform(low=mean-window, high=mean+window))
        # print("mean, offset_linear:", mean, offset)

    if 'fixed' in algo: # e.g. fixed eval schedule
        step_offset = int(self.fps * self.fixed_time_offset)

    return step_offset

  def get_initial_angle(self, algo):
    if 'uniform' in algo:
        # Initialize agent to random orientation [0, 2*pi]
        random_angle = np.random.uniform(0, 2*np.pi)
        agent_angle = np.array([np.cos(random_angle), np.sin(random_angle)]) # Sin and Cos of angle of orientation
    if 'fixed' in algo: # e.g. fixed eval schedule
        agent_angle = np.array([np.cos(self.fixed_angle), np.sin(self.fixed_angle)]) # Sin and Cos of angle of orientation
    return agent_angle

  def diffusion_adjust(self, diffx):
    min_radius = 0.01
    self.data_puffs.loc[:,'radius'] -= min_radius # subtract initial radius
    self.data_puffs.loc[:,'radius'] *= diffx/self.diffusion_max  # adjust 
    self.data_puffs.loc[:,'radius'] += min_radius # add back initial radius
    # Fix other columns
    self.data_puffs['x_minus_radius'] = self.data_puffs.x - self.data_puffs.radius
    self.data_puffs['x_plus_radius'] = self.data_puffs.x + self.data_puffs.radius
    self.data_puffs['y_minus_radius'] = self.data_puffs.y - self.data_puffs.radius
    self.data_puffs['y_plus_radius'] = self.data_puffs.y + self.data_puffs.radius
    self.data_puffs['concentration'] = (min_radius/self.data_puffs.radius)**3

  def reset(self):
    """
    return Gym.Observation
    """
    # print(f'reset() called; self.birthx = {self.birthx}', flush=True)
    self.episode_reward = 0
    self.episode_step = 0 # skip_steps already done during loading
    # Add randomness to start time PER TRIAL!
    self.step_offset = self.get_initial_step_offset(self.time_algo)
    self.t_val = self.t_vals[self.episode_step + self.step_offset] 
    self.t_val_max_episode = self.t_val + 1.0*self.episode_steps_max/self.fps + 1.0
    self.tidx = self.tidxs[self.episode_step + self.step_offset] # Use tidx when possible
    self.tidx_min_episode = self.tidx
    self.tidx_max_episode = self.tidx + self.episode_steps_max*int(100/self.fps) + self.fps 

    # SPEEDUP (subset puffs to those only needed for episode)
    # self.data_puffs = self.data_puffs_all.query('(time > @self.t_val-1) and (time < @self.t_val_max_episode)') # Speeds up queries!
    self.data_puffs = self.data_puffs_all.query('(tidx >= @self.tidx-1) and (tidx <= @self.tidx_max_episode)') # Speeds up queries!
    # print("puff_number_all", self.data_puffs['puff_number'].nunique())
    # Dynamic birthx for each episode
    self.puff_density = 1
    if self.birthx < 0.99:
        puff_density = np.clip(np.random.uniform(low=self.birthx, high=1.0), 0.0, 1.0)
        self.puff_density = puff_density
        # print("puff_density", self.puff_density)
        drop_idxs = self.data_puffs['puff_number'].unique()
        drop_idxs = pd.Series(drop_idxs).sample(frac=(1 - self.puff_density))
        self.data_puffs = self.data_puffs.query("puff_number not in @drop_idxs") # No deep copy being made
        # print("puff_number", self.data_puffs['puff_number'].nunique())
        

    if self.diffusion_min < (self.diffusion_max - 0.01):
        diffx = np.random.uniform(low=self.diffusion_min, high=self.diffusion_max)
        self.diffusion_adjust(diffx)

    # Generalization: Randomly flip plume data across x_axis
    if self.flipping:
        self.flipx = -1.0 if np.random.uniform() > 0.5 else 1.0 
    else:
        self.flipx = 1.0
    # if self.flipx < 0:
    #     self.data_wind = self.data_wind_all.copy(deep=True)
    #     self.data_wind.loc[:,'wind_y'] *= self.flipx
    #     self.data_puffs = self.data_puffs.copy(deep=True)
    #     self.data_puffs.loc[:,'y'] *= self.flipx 
    #     # print(self.data_puffs.shape)
    # else:
    #     self.data_wind = self.data_wind_all

    self.data_wind = self.data_wind_all

    # Initialize agent to random location 
    # self.agent_location = self.get_initial_location(algo='quantile')
    self.agent_location = self.get_initial_location(self.loc_algo)
    self.agent_location_last = self.agent_location
    self.agent_location_init = self.agent_location

    self.stray_distance = self.get_stray_distance()
    self.stray_distance_last = self.stray_distance

    self.agent_angle = self.get_initial_angle(self.angle_algo)
    if self.verbose > 0:
      print("Agent initial location {} and orientation {}".format(self.agent_location, self.agent_angle))
    self.agent_velocity_last = np.array([0, 0])

    # self.wind_ground = self.get_current_wind_xy() # Observe after flip
    self.wind_ground = self.get_current_wind_xy() # Observe after flip
    if self.odor_scaling:
        self.odorx = np.random.uniform(low=0.5, high=1.5) # Odor generalize
    observation = self.sense_environment()
    if self.action_feedback:
        observation = np.concatenate([observation, np.zeros(2)])

    self.found_plume = True if observation[-1] > 0. else False 
    return observation


  def get_oob(self):
    # better restricted bounds    
    # bbox = {'x_min':-2, 'x_max':15, 'y_min':-5, 'y_max':5 }    
    # is_outofbounds = (self.agent_location[0] < bbox['x_min']) or \
    #                  (self.agent_location[0] > bbox['x_max']) or \
    #                  (self.agent_location[1] < bbox['y_min']) or \
    #                  (self.agent_location[1] > bbox['y_max']) 

    is_outofbounds = self.stray_distance > self.stray_max # how far agent can be from closest puff-center
    # if 'switch' in self.dataset: # large perturbations
    #     # bbox = {'x_min':-0.5, 'x_max':10, 'y_min':-3, 'y_max':3 }    

    return is_outofbounds

  def get_current_wind_xy(self):
    # df_idx = self.data_wind.query("time == {}".format(self.t_val)).index[0] # Safer
    df_idx = self.data_wind.query(f"tidx == {self.tidx}").index[0] # Safer
    return self.data_wind.loc[df_idx,['wind_x', 'wind_y']].tolist() # Safer

  # "Transition function"
  def step(self, action):
    """
    return observation, reward, done, info
    """
    self.episode_step += 1 
    self.agent_location_last = self.agent_location
    # Update internal variables
    try:
        self.tidx = self.tidxs[self.episode_step + self.step_offset]
        self.t_val = self.t_vals[self.episode_step + self.step_offset]
    except Exception as ex:
        # Debug case where the env tries to access t_val outside puff_data!
        print("PE_v1 step", ex, self.episode_step, self.step_offset, self.t_val_min, self.t_vals[-5:], self.tidxs[-5:])
        sys.exit(-1)
    
    self.stray_distance_last = self.stray_distance
    self.stray_distance = self.get_stray_distance()
    
    self.wind_ground = self.get_current_wind_xy()
    # print(self.wind_ground)

    # Unpack action
    if self.verbose > 1:
        print("step action:", action, action.shape)
    assert action.shape == (2,)
    if self.squash_action:
        action = (np.tanh(action) + 1)/2
    action = np.clip(action, 0.0, 1.0)
    move_action = action[0] # Typically between [0.0, 1.0]
    turn_action = action[1] # Typically between [0.0, 1.0]
    # print(action)

    # Action: Clip & self.rescale to support more algorithms
    # assert move_action >= 0 and move_action <= 1.0
    # assert turn_action >= 0 and turn_action <= 1.0
    if self.rescale:
        move_action = np.clip(move_action, -1.0, 1.0)
        move_action = (move_action + 1)/2 
        turn_action = np.clip(turn_action, -1.0, 1.0)
        turn_action = (turn_action + 1)/2 

    # Action noise (multiplicative)
    move_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 
    turn_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 

    if self.flipping and self.flipx < 0:
        turn_action = 1 - turn_action

    # Turn/Update orientation and move to new location 
    old_angle_radians = np.angle(self.agent_angle[0] + 1j*self.agent_angle[1], deg=False)
    new_angle_radians = old_angle_radians + self.turn_capacity*self.turnx*(turn_action - 0.5)*self.dt # in radians
    self.agent_angle = [ np.cos(new_angle_radians), np.sin(new_angle_radians) ]    
    assert np.linalg.norm(self.agent_angle) < 1.1

    # New location = old location + agent movement + wind advection
    agent_move_x = self.agent_angle[0]*self.move_capacity*self.movex*move_action*self.dt
    agent_move_y = self.agent_angle[1]*self.move_capacity*self.movex*move_action*self.dt
    wind_drift_x = self.wind_ground[0]*self.dt
    wind_drift_y = self.wind_ground[1]*self.dt
    if self.walking:
        wind_drift_x = wind_drift_y = 0
    self.agent_location = [
      self.agent_location[0] + agent_move_x + wind_drift_x,
      self.agent_location[1] + agent_move_y + wind_drift_y,
    ]
    self.agent_velocity_last = np.array([agent_move_x, agent_move_y])/self.dt # For relative wind calc.

    ### ----------------- End conditions / Is the trial over ----------------- ### 
    is_home = np.linalg.norm(self.agent_location) <= self.homed_radius 
    is_outoftime = self.episode_step >= self.episode_steps_max - 1           
    is_outofbounds = self.get_oob()
    done = bool(is_home or is_outofbounds or is_outoftime)

    # Autocurricula
    # 0.999**1000 = 0.37
    # 0.998**1000 = 0.16
    # 0.997**1000 = 0.05
    # 0.996**1000 = 0.02
    # 0.995**1000 = 0.007
    # 0.99**400 = 0.02
    # 0.95**100 = 0.006
    if is_home and self.auto_movex:
        self.movex = 1 + 0.95*(self.movex - 1)
    if is_home and self.auto_reward:
        self.reward_decay *= 0.995

    # Observation
    observation = self.sense_environment()

    ### ----------------- Reward function ----------------- ### 
    reward = self.rewards['homed'] if is_home else self.rewards['tick']
    if observation[2] <= config.env['odor_threshold'] : # if off plume, more tick penalty
        reward += 5*self.rewards['tick']

    # Reward shaping         
    if is_outofbounds and 'oob' in self.r_shaping:
        # Going OOB should be worse than radial reward shaping
        # OOB Overshooting should be worse!
        oob_penalty = 5*np.linalg.norm(self.agent_location) + self.stray_distance
        oob_penalty *= 2 if self.agent_location[0] < 0 else 1  
        reward -= oob_penalty
         


    # Radial distance decrease at each STEP of episode
    r_radial_step = 0
    if 'step' in self.r_shaping:
        r_radial_step = 5*( np.linalg.norm(self.agent_location_last) - np.linalg.norm(self.agent_location) )
        r_radial_step = min(0, r_radial_step) if observation[2] <= config.env['odor_threshold'] else r_radial_step
        # Multiplier for overshooting source
        if 'overshoot' in self.r_shaping and self.agent_location[0] < 0:
            r_radial_step *= 2 # Both encourage and discourage agent more
        # Additive reward for reducing stray distance from plume
        if ('stray' in self.r_shaping) and (self.stray_distance > self.stray_max/3):
                r_radial_step += 1*(self.stray_distance_last - self.stray_distance)
        reward += r_radial_step * self.reward_decay


    # Walking agent: Metabolic cost: penalize forward movement
    r_metabolic = 0 # for logging
    if self.walking and 'metabolic' in self.r_shaping:
        delta_move = np.linalg.norm(np.array(self.agent_location_last) - np.array(self.agent_location))
        # r_metabolic = -5.*delta_move
        delta_move = 1 if delta_move > 0 else 0
        r_metabolic += self.rewards['tick']*delta_move
        reward += r_metabolic

    # Radial distance decrease at END of episode    
    radial_distance_reward = 0 # keep for logging
    if done and 'end' in self.r_shaping:
        # 1: Radial distance r_decreasease at end of episode
        radial_distance_decrease = ( np.linalg.norm(self.agent_location_init) - np.linalg.norm(self.agent_location) )
        # radial_distance_reward = radial_distance_decrease - np.linalg.norm(self.agent_location)
        # reward += radial_distance_reward 
        # reward -= np.linalg.norm(self.agent_location)
        # end_reward = -np.linalg.norm(self.agent_location)*(1+self.stray_distance) + radial_distance_decrease
        self.stray_distance = self.get_stray_distance()
        # end_reward = -2*self.stray_distance # scale to be comparable with sum_T(r_step)
        end_reward = radial_distance_decrease - self.stray_distance
        reward += end_reward

    r_location = 0 # incorrect, leads to cycling in place
    if 'loc' in self.r_shaping:
        r_location = 1/( 1  + np.linalg.norm(np.array(self.agent_location)) )
        r_location /= (1 + 5*self.stray_distance)
        reward += r_location 

    if 'turn' in self.r_shaping:
        reward -= 0.05*np.abs(2*(turn_action - 0.5))

    if 'move' in self.r_shaping:
        reward -= 0.05*np.abs(move_action)

    if 'found' in self.r_shaping:
        if self.found_plume is False and observation[-1] > 0.:
            # print("found_plume")
            reward += 10
            self.found_plume = True


    reward = reward*self.rewardx # Scale reward for A3C
    
    # Optional/debug info
    done_reason = "HOME" if is_home else \
        "OOB" if is_outofbounds else \
        "OOT" if is_outoftime else \
        "NA"    
    info = {
        't_val':self.t_val, 
        'tidx':self.tidx, 
        'flipx':self.flipx,
        'location':self.agent_location, 
        'location_last':self.agent_location_last, 
        'location_initial':self.agent_location_init, 
        'stray_distance': self.stray_distance,
        'wind_ground': self.wind_ground,
        'angle': self.agent_angle,
        'reward': reward,
        'r_radial_step': r_radial_step,
        # 'reward_decay': self.reward_decay,
        # 'r_radial_ep': radial_distance_reward,
        # 'r_metabolic': r_metabolic,
        'movex': self.movex,
        'done': done_reason if done else None,
        # 'outcomes': self.outcomes,
        'radiusx': self.radiusx,
        }

    if done:
        self.outcomes.append(done_reason)
        if len(self.outcomes) > 10:
            self.outcomes = self.outcomes[1:] # maintain list size

    if done and self.verbose > 0:
        print("{} at (x,y): {}, {} steps w/ reward {}".format( \
            done_reason, 
            self.agent_location, 
            self.episode_step, 
            reward))

    if self.action_feedback:
        # print(observation.shape, action.shape)
        observation = np.concatenate([observation, action])

    if self.flipping and self.flipx < 0:
        observation[1] *= -1.0 # observation: [x, y, o] 

    self.episode_reward += reward
    
    if done:
        info['episode'] = {'r': self.episode_reward }
        info['dataset'] = self.dataset
        info['num_puffs'] = self.data_puffs.puff_number.nunique()
        info['plume_density'] = self.puff_density


    if self.verbose > 0:
        print(observation, reward, done, info)
    return observation, reward, done, info

  def render(self, mode='console'):
    # raise NotImplementedError()
    return

  def close(self):
    del self.data_puffs_all
    del self.data_wind_all
    pass

class PlumeEnvironment_v2(gym.Env):
  """
  Documentation: https://gym.openai.com/docs/#environments
  Plume tracking env v2 
    took out walking and unused, commented-out lines 
  
  """
  def __init__(self, 
    t_val_min=60.00, 
    sim_steps_max=300, # steps
    reset_offset_tmax=30, # seconds; max secs for initial offset from t_val_min
    dataset='constantx5b5',
    move_capacity=2.0, # Max agent speed in m/s
    turn_capacity=6.25*np.pi, # Max agent CW/CCW turn per second
    wind_obsx=1.0, # normalize/divide wind observations by this quantity (move_capacity + wind_max) 
    movex=1.0, # move_max multiplier for tuning
    turnx=1.0, # turn_max multiplier for tuning
    birthx=1.0, # per-episode puff birth rate sparsity minimum
    birthx_max=1.0, # overall odor puff birth rate sparsity max
    env_dt=0.04,
    loc_algo='quantile',
    qvar=1.0, # Variance of init. location; higher = more off-plume initializations
    time_algo='uniform',
    angle_algo='uniform',
    homed_radius=0.2, # meters, at which to end flying episode
    stray_max=2.0, # meters, max distance agent can stray from plume
    wind_rel=True, # Agent senses relative wind speed (not ground speed) # will be overridden by apparent wind if turned on
    auto_movex=False, # simple autocurricula for movex
    auto_reward=False, # simple autocurricula for reward decay
    diff_max=0.8, # teacher curriculum; sets the quantile of init x location 
    diff_min=0.4, # teacher curriculum; sets the quantile of init x location 
    r_shaping=['step', 'oob'], # 'step', 'end'
    rewardx=1.0, # scale reward for e.g. A3C
    squash_action=False, # apply tanh and rescale (useful with PPO) i.e. convert action [0,1] to [-1,1] and then rescale to [0,1]
    walking=False,
    radiusx=1.0, 
    diffusion_min=1.00, 
    diffusion_max=1.00, 
    action_feedback=False,
    flipping=False, # Generalization/reduce training data bias
    odor_scaling=False, # Generalization/reduce training data bias
    obs_noise=0.0, # Multiplicative: Wind & Odor observation noise.
    act_noise=0.0, # Multiplicative: Move & Turn action noise.
    seed=137,
    verbose=0,
    apparent_wind=False,
    apparent_wind_allo=False, # deprecated!
    ):
    super(PlumeEnvironment_v2, self).__init__()

    np.random.seed(seed)    
    self.arguments = locals()
    print("PlumeEnvironment:", self.arguments)
    
    self.verbose = verbose
    self.venv = self
    self.walking = walking
    self.rewardx = rewardx
    self.odor_scaling = odor_scaling
    self.stray_max = stray_max
    self.wind_obsx = wind_obsx
    self.reset_offset_tmax = reset_offset_tmax
    self.action_feedback = action_feedback
    self.qvar = qvar
    self.squash_action = squash_action
    self.obs_noise = obs_noise
    self.act_noise = act_noise
    if self.squash_action:
        print("Squashing actions to 0-1")

    # Fixed evaluation related:
    self.fixed_time_offset = 0.0 # seconds
    self.fixed_angle = 0.0 # downwind
    self.fixed_x = 7.0 
    self.fixed_y = 0.0 # might not work for switch/noisy! 


    # Environment/state variables
    # self.dt = config.env['dt'] 
    self.dt = env_dt # 0.1, 0.2, 0.4, 0.5 sec
    # self.fps = config.env['fps'] # 20/25/50/100 steps/sec
    self.fps = int(1/self.dt)
    # self.sim_fps = 100 # not used
    self.episode_step = 0 # skip_steps done during loading

    # Load simulated data
    self.radiusx = radiusx
    self.birthx = birthx
    self.birthx_max = birthx_max
    self.diffusion_max = diffusion_max # Puff diffusion multiplier (initial)
    self.diffusion_min = diffusion_min # Puff diffusion multiplier (reset-time)
    self.t_val_min = t_val_min
    self.episode_steps_max = sim_steps_max # Short training episodes to gather rewards
    self.t_val_max = self.t_val_min + self.reset_offset_tmax + 1.0*self.episode_steps_max/self.fps + 1.00

    self.set_dataset(dataset)

    # Correction for short simulations
    if self.data_wind.shape[0] < self.episode_steps_max:
      if self.verbose > 0:
        print("Wind data available only up to {} steps".format(self.data_wind.shape[0]))
      self.episode_steps_max = self.data_wind.shape[0]

    # Other initializations -- many redundant, see .reset() 
    # self.agent_location = np.array([1, 0]) # TODO: Smarter
    self.agent_location = None
    self.agent_location_last = self.agent_location
    self.agent_location_init = self.agent_location
    random_angle = np.pi * np.random.uniform(0, 2)
    self.agent_angle_radians = [np.cos(random_angle), np.sin(random_angle)] # Sin and Cos of angle of orientation
    self.step_offset = 0 # random offset per trial in reset()
    self.t_val = self.t_vals[self.episode_step + self.step_offset] 
    self.tidx = self.tidxs[self.episode_step + self.step_offset] 
    self.tidx_min_episode = self.tidx
    self.tidx_max_episode = self.tidx
    self.ambient_wind = None
    self.stray_distance = 0
    self.stray_distance_last = 0
    self.air_velocity = np.array([0, 0]) # Maintain last timestep velocity (in absolute coordinates) for relative sensory observations
    self.episode_reward = 0

    # Generalization & curricula
    self.r_shaping = r_shaping
    # print("Reward Shaping", self.r_shaping)
    self.flipping = flipping 
    self.flipx = 1.0 # flip puffs around x-axis? 
    self.difficulty = diff_max # Curriculum
    self.diff_max = diff_max # Curriculum
    self.diff_min = diff_min # Curriculum
    self.odorx = 1.0 # Doesn't make a difference except when thresholding
    self.turnx = turnx
    self.movex = movex
    self.auto_movex = auto_movex
    self.auto_reward = auto_reward
    self.reward_decay = 1.00
    self.loc_algo = loc_algo
    self.angle_algo = angle_algo
    self.time_algo = time_algo
    assert self.time_algo in ['uniform', 'linear', 'fixed']
    self.outcomes = [] # store outcome last N episodes

    # Constants
    self.wind_rel = wind_rel
    self.turn_capacity = turn_capacity
    self.move_capacity = move_capacity 
    self.arena_bounds = config.env['arena_bounds'] 
    self.homed_radius = homed_radius  # End session if dist(agent - source) < homed_radius
    self.rewards = {
      'tick': -10/self.episode_steps_max,
      'homed': 101.0,
      }

    # Wind Sensing 
    self.apparent_wind = apparent_wind # egocentric app wind = -air velocity (intended direction of movement)

    # Define action and observation spaces
    # Actions:
    # Move [0, 1], with 0.0 = no movement
    # Turn [0, 1], with 0.5 = no turn... maybe change to [-1, 1]
    self.action_space = spaces.Box(low=0, high=+1,
                                        shape=(2,), dtype=np.float32)

    # Observations
    # Wind velocity [-1, 1] * 2, Odor concentration [0, 1]
    obs_dim = 3 if not self.action_feedback else 3+2
    self.observation_space = spaces.Box(low=-1, high=+1,
                                        shape=(obs_dim,), dtype=np.float32)

  def update_env_param(self, params):
      for k,v in params.items():
          setattr(self, k, v)

  def set_dataset(self, dataset):
    self.dataset = dataset
    self.data_puffs_all, self.data_wind_all = load_plume(
        dataset=self.dataset, 
        t_val_min=self.t_val_min, 
        t_val_max=self.t_val_max,
        env_dt=self.dt,
        puff_sparsity=np.clip(self.birthx_max, a_min=0.01, a_max=1.00),
        diffusion_multiplier=self.diffusion_max,
        radius_multiplier=self.radiusx,
        )
    self.data_puffs = self.data_puffs_all.copy() # trim this per episode
    self.data_wind = self.data_wind_all.copy() # trim/flip this per episode
    self.t_vals = self.data_wind['time'].tolist()
    # print("wind: t_val_diff", (self.t_vals[2] - self.t_vals[1]), "env_dt", self.dt)
    t_vals_puffs = self.data_puffs['time'].unique()
    # print("puffs: t_val_diff", (t_vals_puffs[2] - t_vals_puffs[1]), "env_dt", self.dt)
    self.tidxs = self.data_wind['tidx'].tolist()

  def reload_dataset(self):
    self.set_dataset(self.dataset)

  def set_difficulty(self, level, verbose=True): # Curriculum
    """
    Location distance as a form of curriculum learning
    :level: in [0.0, 1.0] with 0.0 being easiest
    """
    if level < 0:
        self.difficulty = self.diff_max
    else:
        level = np.clip(level, 0.0, 1.0)
        self.difficulty = level
    if verbose:
        print("set_difficulty to", self.difficulty)

  def sense_environment(self):
    if (self.verbose > 1) and (self.episode_step >= self.episode_steps_max): # Debug mode
        pprint(vars(self))

    # Wind
    wind_absolute = self.ambient_wind # updated by step(); ambient wind x, y
    
    # Subtract agent velocity to convert to (observed) relative velocity
    if self.wind_rel: 
        wind_absolute = self.ambient_wind - self.air_velocity
    # Use apparent wind (negative of air velocity) for training
    if self.apparent_wind:
        wind_absolute = - self.air_velocity # allocentric apparent wind = negative of air velocity (allocentric)
    if self.verbose > 1:
        print('t_val', self.t_val)
        print('sensed wind (allocentric, before rotating angle by agent direction) ', wind_absolute) 
        
    # Get wind relative angle
    agent_angle_radians = np.angle( self.agent_angle[0] + 1j*self.agent_angle[1], deg=False )
    wind_angle_radians = np.angle( wind_absolute[0] + 1j*wind_absolute[1], deg=False )
    # Add observation noise
    wind_relative_angle_radians = wind_angle_radians - agent_angle_radians 
    wind_observation = [ np.cos(wind_relative_angle_radians), np.sin(wind_relative_angle_radians) ]    
    # Un-normalize wind observation by multiplying by magnitude
    wind_magnitude = np.linalg.norm(np.array( wind_absolute ))/self.wind_obsx
    wind_observation = [ x*wind_magnitude for x in wind_observation ] # convert back to velocity

    if self.verbose > 1:
        print('wind egocentric', wind_observation)
        print('agent_angle', self.agent_angle)

    odor_observation = get_concentration_at_tidx(
        self.data_puffs, self.tidx, self.agent_location[0], self.agent_location[1])
    if self.verbose > 1:
        print('odor_observation', odor_observation)
    if self.odor_scaling:
        odor_observation *= self.odorx # Random scaling to improve generalization 
    odor_observation = 0.0 if odor_observation < config.env['odor_threshold'] else odor_observation
    odor_observation = np.clip(odor_observation, 0.0, 1.0) # clip

    # Return
    observation = np.array(wind_observation + [odor_observation]).astype(np.float32) # per Gym spec
    if self.verbose > 1:
        print('observation', observation)
    return observation

  def get_abunchofpuffs(self, max_samples=300):  
    Z = self.data_puffs.query(f"tidx == {self.tidx}").loc[:,['x','y']]
    Z = Z.sample(n=max_samples, replace=False) if Z.shape[0] > max_samples else Z
    return Z

  def get_stray_distance(self):
    Z = self.get_abunchofpuffs()
    Y = cdist(Z.to_numpy(), np.expand_dims(self.agent_location,axis=0), metric='euclidean')
    try:
        minY = min(Y) 
    except Exception as ex:
        print(f"[PEv2 get_stray_distance] Exception: {ex}, t:{self.t_val:.2f}, tidx:{self.tidx}({self.tidx_min_episode}...{self.tidx_max_episode}), ep_step:{self.episode_step}, {Z}")  
        minY = np.array([0])      
    return minY[0] # return float not float-array

  def get_initial_location(self, algo):
    loc_xy = None
    if 'uniform' in algo:
        loc_xy = np.array([
            2 + np.random.uniform(-1, 1), 
            np.random.uniform(-0.5, 0.5)])


    if 'linear' in algo:
        # TODO
        loc_xy = np.array([
            2 + np.random.uniform(-1, 1), 
            np.random.uniform(-0.5, 0.5)])

    if 'quantile' in algo:
        """ 
        Distance curriculum
        Start the agent at a location with random location with mean and var
        decided by distribution/percentile of puffs 
        """
        q_curriculum = np.random.uniform(self.diff_min, self.diff_max)

        Z = self.get_abunchofpuffs()
        X_pcts = Z['x'].quantile([q_curriculum-0.1, q_curriculum]).to_numpy()
        X_mean, X_var = X_pcts[1], X_pcts[1] - X_pcts[0]
        # print("initial X mean, var, q: ", X_mean, X_var, q_curriculum)
        Y_pcts = Z.query("(x >= (@X_mean - @X_var)) and (x <= (@X_mean + @X_var))")['y'].quantile([0.05,0.5]).to_numpy()
        Y_pcts
        Y_mean, Y_var = Y_pcts[1], min(1, Y_pcts[1] - Y_pcts[0]) # TODO: What was min for?
        # print(Y_mean, Y_var)
        varx = self.qvar 
        loc_xy = np.array([X_mean + varx*X_var*np.random.randn(), 
            Y_mean + varx*Y_var*np.random.randn()]) 

    if 'fixed' in algo:
        loc_xy = np.array( [self.fixed_x, self.fixed_y] )

    return loc_xy

  def get_initial_step_offset(self, algo):
    """ Time curriculum """
    if 'uniform' in algo:
        step_offset = int(self.fps * np.random.uniform(low=0.00, high=self.reset_offset_tmax))

    if 'linear' in algo:
        window = 5 # seconds
        mean = window + self.difficulty*(self.reset_offset_tmax-window)
        step_offset = int(self.fps * np.random.uniform(low=mean-window, high=mean+window))
        # print("mean, offset_linear:", mean, offset)

    if 'fixed' in algo: # e.g. fixed eval schedule
        step_offset = int(self.fps * self.fixed_time_offset)

    return step_offset

  def get_initial_angle(self, algo):
    if 'uniform' in algo:
        # Initialize agent to random orientation [0, 2*pi]
        random_angle = np.random.uniform(0, 2*np.pi)
        agent_angle = np.array([np.cos(random_angle), np.sin(random_angle)]) # Sin and Cos of angle of orientation
    if 'fixed' in algo: # e.g. fixed eval schedule
        agent_angle = np.array([np.cos(self.fixed_angle), np.sin(self.fixed_angle)]) # Sin and Cos of angle of orientation
    return agent_angle

  def diffusion_adjust(self, diffx):
    min_radius = 0.01
    self.data_puffs.loc[:,'radius'] -= min_radius # subtract initial radius
    self.data_puffs.loc[:,'radius'] *= diffx/self.diffusion_max  # adjust 
    self.data_puffs.loc[:,'radius'] += min_radius # add back initial radius
    # Fix other columns
    self.data_puffs['x_minus_radius'] = self.data_puffs.x - self.data_puffs.radius
    self.data_puffs['x_plus_radius'] = self.data_puffs.x + self.data_puffs.radius
    self.data_puffs['y_minus_radius'] = self.data_puffs.y - self.data_puffs.radius
    self.data_puffs['y_plus_radius'] = self.data_puffs.y + self.data_puffs.radius
    self.data_puffs['concentration'] = (min_radius/self.data_puffs.radius)**3

  def reset(self):
    """
    return Gym.Observation
    """
    # print(f'reset() called; self.birthx = {self.birthx}', flush=True)
    self.episode_reward = 0
    self.episode_step = 0 # skip_steps already done during loading
    # Add randomness to start time PER TRIAL!
    self.step_offset = self.get_initial_step_offset(self.time_algo)
    self.t_val = self.t_vals[self.episode_step + self.step_offset] 
    self.t_val_max_episode = self.t_val + 1.0*self.episode_steps_max/self.fps + 1.0
    self.tidx = self.tidxs[self.episode_step + self.step_offset] # Use tidx when possible
    self.tidx_min_episode = self.tidx
    self.tidx_max_episode = self.tidx + self.episode_steps_max*int(100/self.fps) + self.fps 

    # SPEEDUP (subset puffs to those only needed for episode)
    # self.data_puffs = self.data_puffs_all.query('(time > @self.t_val-1) and (time < @self.t_val_max_episode)') # Speeds up queries!
    self.data_puffs = self.data_puffs_all.query('(tidx >= @self.tidx-1) and (tidx <= @self.tidx_max_episode)') # Speeds up queries!
    # print("puff_number_all", self.data_puffs['puff_number'].nunique())
    # Dynamic birthx for each episode
    self.puff_density = 1
    if self.birthx < 0.99:
        puff_density = np.clip(np.random.uniform(low=self.birthx, high=1.0), 0.0, 1.0)
        self.puff_density = puff_density
        # print("puff_density", self.puff_density)
        drop_idxs = self.data_puffs['puff_number'].unique()
        drop_idxs = pd.Series(drop_idxs).sample(frac=(1 - self.puff_density))
        self.data_puffs = self.data_puffs.query("puff_number not in @drop_idxs") # No deep copy being made
        # print("puff_number", self.data_puffs['puff_number'].nunique())
        

    if self.diffusion_min < (self.diffusion_max - 0.01):
        diffx = np.random.uniform(low=self.diffusion_min, high=self.diffusion_max)
        self.diffusion_adjust(diffx)

    # Generalization: Randomly flip plume data across x_axis
    if self.flipping:
        self.flipx = -1.0 if np.random.uniform() > 0.5 else 1.0 
    else:
        self.flipx = 1.0

    self.data_wind = self.data_wind_all

    # Initialize agent to random location 
    self.agent_location = self.get_initial_location(self.loc_algo)
    self.agent_location_last = self.agent_location
    self.agent_location_init = self.agent_location

    self.stray_distance = self.get_stray_distance()
    self.stray_distance_last = self.stray_distance

    self.agent_angle = self.get_initial_angle(self.angle_algo)
    if self.verbose > 0:
      print("Agent initial location {} and orientation {}".format(self.agent_location, self.agent_angle))
    self.air_velocity = np.array([0, 0])
    
    self.ambient_wind = self.get_current_wind_xy() # Observe after flip
    if self.odor_scaling:
        self.odorx = np.random.uniform(low=0.5, high=1.5) # Odor generalize
    observation = self.sense_environment()
    if self.action_feedback:
        observation = np.concatenate([observation, np.zeros(2)])

    self.found_plume = True if observation[-1] > 0. else False 
    return observation


  def get_oob(self):
    # better restricted bounds    
    is_outofbounds = self.stray_distance > self.stray_max # how far agent can be from closest puff-center
    return is_outofbounds

  def get_current_wind_xy(self):
    # df_idx = self.data_wind.query("time == {}".format(self.t_val)).index[0] # Safer
    df_idx = self.data_wind.query(f"tidx == {self.tidx}").index[0] # Safer
    return self.data_wind.loc[df_idx,['wind_x', 'wind_y']].tolist() # Safer

  # "Transition function"
  def step(self, action):
    """
    return observation, reward, done, info
    """
    self.episode_step += 1 
    self.agent_location_last = self.agent_location
    # Update internal variables
    try:
        self.tidx = self.tidxs[self.episode_step + self.step_offset]
        self.t_val = self.t_vals[self.episode_step + self.step_offset]
    except Exception as ex:
        # Debug case where the env tries to access t_val outside puff_data!
        print("[PEv2 step]",ex, self.episode_step, self.step_offset, self.t_val_min, self.t_vals[-5:], self.tidxs[-5:])
        sys.exit(-1)
    
    self.stray_distance_last = self.stray_distance
    self.stray_distance = self.get_stray_distance()
    
    self.ambient_wind = self.get_current_wind_xy()
    # print(self.ambient_wind)

    # Unpack action
    if self.verbose > 1:
        print("step action:", action, action.shape)

    if self.squash_action: # always true in training and eval. Bkw compt for Sat's older logs in visualization scripts... Not touching this yet.
        action = (np.tanh(action) + 1)/2
    action = np.clip(action, 0.0, 1.0)
    move_action = action[0] # Move [0, 1], with 0.0 = no movement
    turn_action = action[1] # # Turn [0, 1], with 0.5 = no turn... maybe change to [-1, 1]

    # Action noise (multiplicative)
    move_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 
    turn_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 

    if self.flipping and self.flipx < 0:
        turn_action = 1 - turn_action

    # Turn/Update orientation and move to new location 
    old_angle_radians = np.angle(self.agent_angle[0] + 1j*self.agent_angle[1], deg=False)
    new_angle_radians = old_angle_radians + self.turn_capacity*self.turnx*(turn_action - 0.5)*self.dt # in radians; (Turn~[0, 1], with 0.5 = no turn, <0.5 turn cw, >0.5 turn ccw)
    self.agent_angle = [ np.cos(new_angle_radians), np.sin(new_angle_radians) ]    
    assert np.linalg.norm(self.agent_angle) < 1.1

    # New location = old location + agent movement + wind advection
    agent_move_x = self.agent_angle[0]*self.move_capacity*self.movex*move_action*self.dt
    agent_move_y = self.agent_angle[1]*self.move_capacity*self.movex*move_action*self.dt
    wind_drift_x = self.ambient_wind[0]*self.dt
    wind_drift_y = self.ambient_wind[1]*self.dt

    self.agent_location = [
      self.agent_location[0] + agent_move_x + wind_drift_x,
      self.agent_location[1] + agent_move_y + wind_drift_y,
    ]
    self.air_velocity = np.array([agent_move_x, agent_move_y])/self.dt
    self.ground_velocity = np.array(self.agent_location_last - self.agent_location)/self.dt
    ### ----------------- End conditions / Is the trial over ----------------- ### 
    is_home = np.linalg.norm(self.agent_location) <= self.homed_radius 
    is_outoftime = self.episode_step >= self.episode_steps_max - 1           
    is_outofbounds = self.get_oob()
    done = bool(is_home or is_outofbounds or is_outoftime)

    # Autocurricula
    # 0.999**1000 = 0.37
    # 0.998**1000 = 0.16
    # 0.997**1000 = 0.05
    # 0.996**1000 = 0.02
    # 0.995**1000 = 0.007
    # 0.99**400 = 0.02
    # 0.95**100 = 0.006
    if is_home and self.auto_movex:
        self.movex = 1 + 0.95*(self.movex - 1)
    if is_home and self.auto_reward:
        self.reward_decay *= 0.995

    # Observation
    observation = self.sense_environment()

    ### ----------------- Reward function ----------------- ### 
    reward = self.rewards['homed'] if is_home else self.rewards['tick']
    if observation[2] <= config.env['odor_threshold'] : # if off plume, more tick penalty
        reward += 5*self.rewards['tick']

    # Reward shaping         
    if is_outofbounds and 'oob' in self.r_shaping:
        # Going OOB should be worse than radial reward shaping
        # OOB Overshooting should be worse!
        oob_penalty = 5*np.linalg.norm(self.agent_location) + self.stray_distance
        oob_penalty *= 2 if self.agent_location[0] < 0 else 1  
        reward -= oob_penalty


    # Radial distance decrease at each STEP of episode
    r_radial_step = 0
    if 'step' in self.r_shaping:
        r_radial_step = 5*( np.linalg.norm(self.agent_location_last) - np.linalg.norm(self.agent_location) )
        r_radial_step = min(0, r_radial_step) if observation[2] <= config.env['odor_threshold'] else r_radial_step
        # Multiplier for overshooting source
        if 'overshoot' in self.r_shaping and self.agent_location[0] < 0:
            r_radial_step *= 2 # Both encourage and discourage agent more
        # Additive reward for reducing stray distance from plume
        if ('stray' in self.r_shaping) and (self.stray_distance > self.stray_max/3):
                r_radial_step += 1*(self.stray_distance_last - self.stray_distance)
        reward += r_radial_step * self.reward_decay

    # Radial distance decrease at END of episode    
    radial_distance_reward = 0 # keep for logging
    if done and 'end' in self.r_shaping:
        # 1: Radial distance r_decreasease at end of episode
        radial_distance_decrease = ( np.linalg.norm(self.agent_location_init) - np.linalg.norm(self.agent_location) )
        self.stray_distance = self.get_stray_distance()
        end_reward = radial_distance_decrease - self.stray_distance
        reward += end_reward

    r_location = 0 # incorrect, leads to cycling in place
    if 'loc' in self.r_shaping:
        r_location = 1/( 1  + np.linalg.norm(np.array(self.agent_location)) )
        r_location /= (1 + 5*self.stray_distance)
        reward += r_location 

    if 'turn' in self.r_shaping:
        reward -= 0.05*np.abs(2*(turn_action - 0.5))

    if 'move' in self.r_shaping:
        reward -= 0.05*np.abs(move_action)

    if 'found' in self.r_shaping:
        if self.found_plume is False and observation[-1] > 0.:
            # print("found_plume")
            reward += 10
            self.found_plume = True


    reward = reward*self.rewardx # Scale reward for A3C
    
    # Optional/debug info
    done_reason = "HOME" if is_home else \
        "OOB" if is_outofbounds else \
        "OOT" if is_outoftime else \
        "NA"    
    info = {
        't_val':self.t_val, 
        'tidx':self.tidx, 
        'flipx':self.flipx,
        'location':self.agent_location, 
        'location_last':self.agent_location_last, 
        'location_initial':self.agent_location_init, 
        'stray_distance': self.stray_distance,
        'ambient_wind': self.ambient_wind,
        'angle': self.agent_angle,
        'reward': reward,
        'r_radial_step': r_radial_step,
        'movex': self.movex,
        'done': done_reason if done else None,
        'radiusx': self.radiusx,
        }

    if done:
        self.outcomes.append(done_reason)
        if len(self.outcomes) > 10:
            self.outcomes = self.outcomes[1:] # maintain list size

    if done and self.verbose > 0:
        print("{} at (x,y): {}, {} steps w/ reward {}".format( \
            done_reason, 
            self.agent_location, 
            self.episode_step, 
            reward))

    if self.action_feedback:
        # print(observation.shape, action.shape)
        observation = np.concatenate([observation, action])

    if self.flipping and self.flipx < 0:
        observation[1] *= -1.0 # observation: [x, y, o] 

    self.episode_reward += reward
    
    if done:
        info['episode'] = {'r': self.episode_reward }
        info['dataset'] = self.dataset
        info['num_puffs'] = self.data_puffs.puff_number.nunique()
        info['plume_density'] = self.puff_density


    if self.verbose > 0:
        print(observation, reward, done, info)
    return observation, reward, done, info

  def render(self, mode='console'):
    # raise NotImplementedError()
    return

  def close(self):
    del self.data_puffs_all
    del self.data_wind_all
    pass

class PlumeEnvironment_v3(PlumeEnvironment_v2):
    def __init__(self, visual_feedback=False, flip_ventral_optic_flow=False, **kwargs):
        super(PlumeEnvironment_v3, self).__init__(**kwargs)
        self.flip_ventral_optic_flow = flip_ventral_optic_flow
        if self.verbose > 0:
            print("PlumeEnvironment_v3")
            print("visual_feedback", visual_feedback)
            print("flip_ventral_optic_flow", flip_ventral_optic_flow)
        self.visual_feedback = visual_feedback
        self.ground_velocity = np.array([0, 0]) # for egocentric course direction calculation
        if self.visual_feedback:
            self.observation_space = spaces.Box(low=-1, high=+1,
                                        shape=(7,), dtype=np.float32) # [wind x, y, odor, head direction x, y, course direction x, y]
        else:
            self.observation_space = spaces.Box(low=-1, high=+1,
                                    shape=(3,), dtype=np.float32) # [(apparent/ambient) wind x, y, odor]
        # convert obs_noise to radians
        if self.obs_noise:
            self.obs_noise = np.deg2rad(self.obs_noise)

    def sense_environment(self):
        '''
        Return an array with [wind x, y, odor, allocentric head direction x, y, egocentric course direction x, y]
        '''
        # Get an array with [wind x, y, odor]
            # Wind can either be relative wind or apparent wind, depending on the setting
        observation = super(PlumeEnvironment_v3, self).sense_environment()
        # Visual feedback
        if self.visual_feedback:
            # head direction
            allocentric_head_direction_radian = np.angle(self.agent_angle[0] + 1j*self.agent_angle[1], deg=False)
            # course direction
            allocentric_course_direction_radian = np.angle(self.ground_velocity[0] + 1j*self.ground_velocity[1], deg=False)
            egocentric_course_direction_radian = allocentric_course_direction_radian - allocentric_head_direction_radian # leftward positive - standard CWW convention
            if self.flip_ventral_optic_flow:
                egocentric_course_direction_radian = allocentric_head_direction_radian - allocentric_course_direction_radian # rightward positive - for eval to see the behavioral impact of flipping course direction perception.
            if self.obs_noise:
                allocentric_head_direction_radian += np.random.normal(0, self.obs_noise)
                egocentric_course_direction_radian += np.random.normal(0, self.obs_noise)
                apparent_wind_radian = np.angle(observation[0] + 1j*observation[1], deg=False) + np.random.normal(0, self.obs_noise)
                observation[:2] = [np.cos(apparent_wind_radian), np.sin(apparent_wind_radian)]
            observation = np.append(observation, [np.cos(allocentric_head_direction_radian), np.sin(allocentric_head_direction_radian), np.cos(egocentric_course_direction_radian), np.sin(egocentric_course_direction_radian)])
        if self.verbose > 1:
            print('observation', observation)
        return observation
    
    def reset(self):
        # Return an array with [wind x, y, odor, air vel x, y, egocentric course direction x, y]
            # Wind can either be relative wind or apparent wind, depending on the setting
        observation = super(PlumeEnvironment_v3, self).reset()
        if len(observation) == 7:
            observation[5:] = 0 # course direction to 0
        return observation
    
    def step(self, action):
        """
        return observation, reward, done, info
        same as v2, but with ground velco and air velco as attributes
        """
        self.episode_step += 1 
        self.agent_location_last = self.agent_location
        # Update internal variables
        try:
            self.tidx = self.tidxs[self.episode_step + self.step_offset]
            self.t_val = self.t_vals[self.episode_step + self.step_offset]
        except Exception as ex:
            # Debug case where the env tries to access t_val outside puff_data!
            print("[PEv3 step]", ex, self.episode_step, self.step_offset, self.t_val_min, self.t_vals[-5:], self.tidxs[-5:])
            sys.exit(-1)
        
        self.stray_distance_last = self.stray_distance
        self.stray_distance = self.get_stray_distance()
        self.ambient_wind = self.get_current_wind_xy()

        # Handle action 
        if self.verbose > 1:
            print("step action:", action, action.shape)
        if self.squash_action: # always true in training and eval. Bkw compt for Sat's older logs in visualization scripts... Not touching this yet.
            action = (np.tanh(action) + 1)/2
        action = np.clip(action, 0.0, 1.0)
        move_action = action[0] # Move [0, 1], with 0.0 = no movement
        turn_action = action[1] # # Turn [0, 1], with 0.5 = no turn... maybe change to [-1, 1]
        # Action noise (multiplicative)
        move_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 
        turn_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 
        
        # Flipping arena? 
        if self.flipping and self.flipx < 0:
            turn_action = 1 - turn_action

        # Consequences of Actions
        # Turn/Update orientation and move to new location 
        old_angle_radians = np.angle(self.agent_angle[0] + 1j*self.agent_angle[1], deg=False)
        new_angle_radians = old_angle_radians + self.turn_capacity*self.turnx*(turn_action - 0.5)*self.dt # in radians; (Turn~[0, 1], with 0.5 = no turn, <0.5 turn cw, >0.5 turn ccw)
        self.agent_angle = [ np.cos(new_angle_radians), np.sin(new_angle_radians) ]    
        assert np.linalg.norm(self.agent_angle) < 1.1
        # New location = old location + agent movement + wind advection
        agent_move_x = self.agent_angle[0]*self.move_capacity*self.movex*move_action*self.dt
        agent_move_y = self.agent_angle[1]*self.move_capacity*self.movex*move_action*self.dt
        wind_drift_x = self.ambient_wind[0]*self.dt
        wind_drift_y = self.ambient_wind[1]*self.dt
        self.agent_location = [
        self.agent_location[0] + agent_move_x + wind_drift_x,
        self.agent_location[1] + agent_move_y + wind_drift_y,
        ]
        # Air and ground velocity
        self.air_velocity = np.array([agent_move_x, agent_move_y])/self.dt # Rel_wind = Amb_wind - Air_vel
        self.ground_velocity = (np.array(self.agent_location) - self.agent_location_last)/self.dt
        
        ### ----------------- End conditions / Is the trial over ----------------- ### 
        is_home = np.linalg.norm(self.agent_location) <= self.homed_radius 
        is_outoftime = self.episode_step >= self.episode_steps_max - 1           
        is_outofbounds = self.get_oob()
        done = bool(is_home or is_outofbounds or is_outoftime)

        # If has not ended, make new observations
        observation = self.sense_environment()

        ### ----------------- Reward function ----------------- ### 
        reward = self.rewards['homed'] if is_home else self.rewards['tick']
        if observation[2] <= config.env['odor_threshold'] : # if off plume, more tick penalty
            reward += 5*self.rewards['tick']
        # Reward shaping         
        if is_outofbounds and 'oob' in self.r_shaping:
            # Going OOB should be worse than radial reward shaping
            # OOB Overshooting should be worse!
            oob_penalty = 5*np.linalg.norm(self.agent_location) + self.stray_distance
            oob_penalty *= 2 if self.agent_location[0] < 0 else 1  
            reward -= oob_penalty
        # Radial distance decrease at each STEP of episode
        r_radial_step = 0
        if 'step' in self.r_shaping:
            r_radial_step = 5*( np.linalg.norm(self.agent_location_last) - np.linalg.norm(self.agent_location) )
            r_radial_step = min(0, r_radial_step) if observation[2] <= config.env['odor_threshold'] else r_radial_step
            # Multiplier for overshooting source
            if 'overshoot' in self.r_shaping and self.agent_location[0] < 0:
                r_radial_step *= 2 # Both encourage and discourage agent more
            # Additive reward for reducing stray distance from plume
            if ('stray' in self.r_shaping) and (self.stray_distance > self.stray_max/3):
                    r_radial_step += 1*(self.stray_distance_last - self.stray_distance)
            reward += r_radial_step * self.reward_decay

        # Radial distance decrease at END of episode    
        radial_distance_reward = 0 # keep for logging
        if done and 'end' in self.r_shaping:
            # 1: Radial distance r_decreasease at end of episode
            radial_distance_decrease = ( np.linalg.norm(self.agent_location_init) - np.linalg.norm(self.agent_location) )
            self.stray_distance = self.get_stray_distance()
            end_reward = radial_distance_decrease - self.stray_distance
            reward += end_reward

        r_location = 0 # incorrect, leads to cycling in place
        if 'loc' in self.r_shaping:
            r_location = 1/( 1  + np.linalg.norm(np.array(self.agent_location)) )
            r_location /= (1 + 5*self.stray_distance)
            reward += r_location 

        if 'turn' in self.r_shaping:
            reward -= 0.05*np.abs(2*(turn_action - 0.5))

        if 'move' in self.r_shaping:
            reward -= 0.05*np.abs(move_action)

        if 'found' in self.r_shaping:
            if self.found_plume is False and observation[-1] > 0.:
                # print("found_plume")
                reward += 10
                self.found_plume = True


        reward = reward*self.rewardx # Scale reward for A3C
        
        # Optional/debug info
        done_reason = "HOME" if is_home else \
            "OOB" if is_outofbounds else \
            "OOT" if is_outoftime else \
            "NA"    
        info = {
            't_val':self.t_val, 
            'tidx':self.tidx, 
            'flipx':self.flipx,
            'location':self.agent_location, 
            'location_last':self.agent_location_last, 
            'location_initial':self.agent_location_init, 
            'stray_distance': self.stray_distance,
            'ambient_wind': self.ambient_wind,
            'angle': self.agent_angle,
            'reward': reward,
            'r_radial_step': r_radial_step,
            'movex': self.movex,
            'done': done_reason if done else None,
            'radiusx': self.radiusx,
            'air_velcity': self.air_velocity,
            'ground_velocity': self.ground_velocity,
            'wind_obs': observation[:2],
            'odor_obs': observation[2],
            'allocentric_head_direction': observation[3:5],
            'egocentric_course_direction': observation[5:7] # within SubprocVecEnv, unnormalized obs. Later nomalized in VecPyTorch
            }

        if done:
            self.outcomes.append(done_reason)
            if len(self.outcomes) > 10:
                self.outcomes = self.outcomes[1:] # maintain list size

        if done and self.verbose > 0:
            print("{} at (x,y): {}, {} steps w/ reward {}".format( \
                done_reason, 
                self.agent_location, 
                self.episode_step, 
                reward))

        if self.action_feedback:
            # print(observation.shape, action.shape)
            observation = np.concatenate([observation, action])

        if self.flipping and self.flipx < 0:
            observation[1] *= -1.0 # observation: [x, y, o] 

        self.episode_reward += reward
        
        if done:
            info['episode'] = {'r': self.episode_reward }
            info['dataset'] = self.dataset
            info['num_puffs'] = self.data_puffs.puff_number.nunique()
            info['plume_density'] = self.puff_density


        if self.verbose > 0:
            print(observation, reward, done, info)
        return observation, reward, done, info

        
def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  args=None,
                  num_frame_stack=None, 
                  **raw_kwargs # raw because it may contain parameters that are already in args, in which case overwrite args. 
                  ):
    envs = []
    if raw_kwargs:
        if 'dataset' in raw_kwargs: # curriculum learning which multiple datasets for different types of environments 
            for env_idx in range(len(raw_kwargs['dataset'])):
                processed_kwargs = {} 
                for k, v in raw_kwargs.items():
                    # check if v is iterable. If so, it's a curriculum variable and will be treated as such
                    if isinstance(v, (list, tuple)):
                        processed_kwargs[k] = v[env_idx]
                    else:
                        processed_kwargs[k] = v
                args_copy = copy.deepcopy(args) # need to copy args to avoid overwriting the same object. make_env returns a function without immediately creating the env
                for k,v in processed_kwargs.items():
                    setattr(args_copy, k, v)
                for i in range(num_processes):
                    envs.append(make_env(env_name, seed, i, log_dir, allow_early_resets, args_copy))
        else: # not sure if kwargs will ever be used outside of curriculum learning... ignore this portion for now. 
            args_copy = copy.deepcopy(args)
            for k,v in raw_kwargs.items():
                setattr(args_copy, k, v)
            for i in range(num_processes):
                envs.append(make_env(env_name, seed, i, log_dir, allow_early_resets, args_copy))
    else:
        for i in range(num_processes):
            envs.append(make_env(env_name, seed, i, log_dir, allow_early_resets, args))

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
        envs.num_envs = num_processes
        envs.deploy(range(num_processes))
    else:
        envs = DummyVecEnv(envs)
        
    if len(envs.observation_space.shape) == 1:
        if raw_kwargs:
            if 'vecNormalize_pkl_file' in raw_kwargs:
                envs = VecNormalize.load(raw_kwargs['vecNormalize_pkl_file'], envs)
                if 'eval' in raw_kwargs: # if eval is true, then set envs to eval mode to stop updating the statistics and reward norm
                    if raw_kwargs['eval']:
                        envs.eval()
                    else:
                        envs.train()
                else: # if not specified, assume it's training. May be relevant when using checkpoint
                    envs.train()
            else:
                if gamma is None:
                    envs = VecNormalize(envs, ret=False)
                else:
                    envs = VecNormalize(envs, gamma=gamma)
        else:
            if gamma is None:
                envs = VecNormalize(envs, ret=False)
            else:
                envs = VecNormalize(envs, gamma=gamma) # type(envs.action_space) = <class 'gym.spaces.box.Box'> # gamma, reward discount factor = 0.99 always 
        
    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs

def make_env(env_id, seed, rank, log_dir, allow_early_resets, args=None):
    def _thunk():
        if args.recurrent_policy or (args.stacking == 0):
            if 'apparent_wind' in args:
                if 'visual_feedback' not in args:
                    env = PlumeEnvironment_v2(
                        dataset=args.dataset,
                        birthx=args.birthx, 
                        qvar=args.qvar,
                        diff_max=args.diff_max,
                        diff_min=args.diff_min,
                        reset_offset_tmax=args.reset_offset_tmax,
                        t_val_min=args.t_val_min,
                        turnx=args.turnx,
                        movex=args.movex,
                        birthx_max=args.birthx_max,
                        env_dt=args.env_dt,
                        loc_algo=args.loc_algo,
                        time_algo=args.time_algo,
                        auto_movex=args.auto_movex,
                        auto_reward=args.auto_reward,
                        walking=args.walking,
                        radiusx=args.radiusx,
                        r_shaping=args.r_shaping,
                        wind_rel=args.wind_rel,
                        action_feedback=args.action_feedback,
                        squash_action=args.squash_action,
                        flipping=args.flipping,
                        odor_scaling=args.odor_scaling,
                        stray_max=args.stray_max,
                        obs_noise=args.obs_noise,
                        act_noise=args.act_noise,
                        seed=args.seed,
                        apparent_wind=args.apparent_wind
                        )
                else:
                    env = PlumeEnvironment_v3(
                        dataset=args.dataset,
                        birthx=args.birthx, 
                        qvar=args.qvar,
                        diff_max=args.diff_max,
                        diff_min=args.diff_min,
                        reset_offset_tmax=args.reset_offset_tmax,
                        t_val_min=args.t_val_min,
                        turnx=args.turnx,
                        movex=args.movex,
                        birthx_max=args.birthx_max,
                        env_dt=args.env_dt,
                        loc_algo=args.loc_algo,
                        time_algo=args.time_algo,
                        auto_movex=args.auto_movex,
                        auto_reward=args.auto_reward,
                        walking=args.walking,
                        radiusx=args.radiusx,
                        r_shaping=args.r_shaping,
                        wind_rel=args.wind_rel,
                        action_feedback=args.action_feedback,
                        squash_action=args.squash_action,
                        flipping=args.flipping,
                        odor_scaling=args.odor_scaling,
                        stray_max=args.stray_max,
                        obs_noise=args.obs_noise,
                        act_noise=args.act_noise,
                        seed=args.seed,
                        apparent_wind=args.apparent_wind,
                        visual_feedback=args.visual_feedback,
                        flip_ventral_optic_flow=args.flip_ventral_optic_flow
                        )
            else:
                # bkw compat before cleaning up TC hack. Useful when evalCli
                env = PlumeEnvironment(
                    dataset=args.dataset,
                    turnx=args.turnx,
                    movex=args.movex,
                    birthx=args.birthx,
                    birthx_max=args.birthx_max,
                    env_dt=args.env_dt,
                    loc_algo=args.loc_algo,
                    time_algo=args.time_algo,
                    diff_max=args.diff_max,
                    diff_min=args.diff_min,
                    auto_movex=args.auto_movex,
                    auto_reward=args.auto_reward,
                    walking=args.walking,
                    radiusx=args.radiusx,
                    r_shaping=args.r_shaping,
                    wind_rel=args.wind_rel,
                    action_feedback=args.action_feedback,
                    squash_action=args.squash_action,
                    flipping=args.flipping,
                    odor_scaling=args.odor_scaling,
                    qvar=args.qvar,
                    stray_max=args.stray_max,
                    obs_noise=args.obs_noise,
                    act_noise=args.act_noise,
                    seed=args.seed
                    )

        env.seed(seed + rank)        

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])
        return env

    return _thunk

def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    """
    Flatten observations, depending on the observation space.

    :param obs: observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return: flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
    else:
        return np.stack(obs)


def _worker(
    remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper
) -> None:
    # Import here to avoid a circular import
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                # no longer resets if done. Moved to step_async_wait in my SubprocVecEnv
                observation, reward, done, info = env.step(data)
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            print("Worker got EOFError", flush=True, file=sys.stderr)
            break


class SubprocVecEnv(SubprocVecEnv_):
    # SubprocVecEnv_ is from gym, as seen above in imports
    # my version that supports running only a subset of envs 
    # also manages swapping of environments 
    # also manages current MAX delta wind direction
    
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        # inherit __init__ from Gym SubprocVecEnv
        SubprocVecEnv_.__init__(self, env_fns, start_method)
        # lists of remotes and processes that are currently deployed
        self.deployed_remotes = list(self.remotes)
        self.deployed_processes = self.processes
        self.wind_directions = 1 # the MAX number of directions of wind in the environment
        self.remote_directory = {} # key: index of remote, value: dataset name and deployment status
        available_datasets = self.get_attr_all_envs('dataset') # list of all available datasets 
        
        for i, ds in enumerate(available_datasets):
            self.remote_directory[i] = {'dataset': ds, 'deployed': True, 'wind_direction': self.ds2wind(ds)}
        
    def update_wind_direction(self, new_max_wind_direction: int):
        # only support 1, 2, 3 which correspons to constant, switch, noisy
        assert new_max_wind_direction <= 3 
        assert new_max_wind_direction > 0 
        self.wind_directions = new_max_wind_direction
    
    def sample_wind_direction(self):
        wind_dir = np.random.randint(1, self.wind_directions+1) # +1 because randint is end-exclusive
        return wind_dir
    
    def refresh_deployment_status(self):
        # check all processes and update deployment status
        for i, r in enumerate(self.remotes):
            if r in self.deployed_remotes:
                self.remote_directory[i]['deployed'] = True
            else:
                self.remote_directory[i]['deployed'] = False
        
    def deploy(self, indices: VecEnvIndices = None) -> None:
        """
        Deploy envs in subprocesses.
        :param indices: refers to indices of envs.
        """
        indices = self._get_indices(indices)
        self.deployed_remotes = [self.remotes[i] for i in indices]
        self.deployed_processes = [self.processes[i] for i in indices]
        assert len(indices) == self.num_envs, "cannot deploy more envs than the predetermined num_processes "
        self.refresh_deployment_status()
        
    def ds2wind(self, ds):
        # translate dataset name to number of changes in wind direction
        # input: dataset name
        # output: number of changes in wind direction (3 as in the agent may encounter up to 3 different wind directions)
        # TODO: add switch condition... Skip for now. Now sure if concrete difference from nosiy.. 
        if 'noisy' in ds:
            return 3
        elif 'constant' in ds:
            return 1
        elif 'switch' in ds:
            return 2
        else:
            raise NotImplementedError
        
    def swap(self, idx: int, idx_replacement_item: int) -> None:
        """
        Swap envs in subprocesses.
        :param indices: refers to indices of envs.
        """
        self.deployed_remotes[idx] = self.remotes[idx_replacement_item]
        self.deployed_processes[idx] = self.processes[idx_replacement_item]
        self.refresh_deployment_status()
        
    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.deployed_remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def check_remote_sanity(self):
        # check if there are any duplicated remotes that are deployed
        # actually won't be a problem since iterates over self.deployed_remotes
        assert len(self.deployed_remotes) == len(set(self.deployed_processes)), "duplicated remotes"
    
    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.deployed_remotes]
        self.waiting = False
        swapped = False
        obs, rews, dones, infos = zip(*results)
        for i, d in enumerate(dones): 
            if d:
                # log the final observation
                infos[i]["terminal_observation"] = obs[i]
                # sample a new wind condition
                new_wind_direction = self.sample_wind_direction()
                # print('[DEBUG] sampled wind direction:', new_wind_direction)
                current_wind_direction = self.ds2wind(self.get_attr('dataset', i)[0])
                # print('[DEBUG] current wind direction:', current_wind_direction)
                # if wind condtion changed, then swap
                if new_wind_direction != current_wind_direction:
                    # print(f"[DEBUG] new wind dir selected... pre swap {self.get_attr('dataset')}")
                    # check the remote_directory to swap with an undeployed env with the condition of interest
                    for remote_idx, status in self.remote_directory.items():
                        if status['deployed'] == False and status['wind_direction'] == new_wind_direction:
                            self.swap(i, remote_idx)
                            # print(f"[DEBUG] new wind dir selected... post swap {self.get_attr('dataset')}")
                            swapped = True
                            break
                
                # update the newest observation
                list(obs)[i] = self.reset_deployed_at(i)
                obs = tuple(obs)
        
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        for idx, remote in enumerate(self.deployed_remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.deployed_remotes]

    def reset(self) -> VecEnvObs:
        # only reset deployed envs
        for remote in self.deployed_remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.deployed_remotes]
        return _flatten_obs(obs, self.observation_space)
    
    def reset_deployed_at(self, indices: VecEnvIndices = None) -> VecEnvObs:
        # reset only envs at indices
        indices = self._get_indices(indices)
        for i in indices:
            self.deployed_remotes[i].send(("reset", None))
        obs = [self.deployed_remotes[i].recv() for i in indices]
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        # unchanged. Not sure when it is called automatically. Never called manually
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[np.ndarray]:
        # unchanged. Not sure when it is called automatically. Never called manually
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(("render", "rgb_array"))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from the DEPLOYED environments."""
        target_remotes = self._get_target_remotes(indices, deployed=True)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]
    
    def get_attr_all_envs(self, attr_name: str) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        indices = range(len(self.remotes))
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        raise NotImplementedError("set_attr is deprecated... can set attr but does not take effect. need to set via env_method!")
        target_remotes = self._get_target_remotes(indices, deployed=True)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()
            
    def set_attr_all_env(self, attr_name: str, value: Any) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        raise NotImplementedError("set_attr_all_env is deprecated... can set attr but does not take effect. need to set via env_method!")
        indices = range(len(self.remotes))
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments. Apply method to deployed envs only"""
        target_remotes = self._get_target_remotes(indices) # get indices of deployed envs
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]
    
    def env_method_apply_to_all(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments. Apply method to all environments"""
        indices = range(len(self.remotes))
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices, deployed: bool = False) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        if deployed:
            return [self.deployed_remotes[i] for i in indices]
        return [self.remotes[i] for i in indices]


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    

class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True
    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs
    def train(self):
        self.training = True
    def eval(self):
        self.training = False

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
