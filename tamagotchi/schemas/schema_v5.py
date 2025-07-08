import datajoint as dj
import hashlib
import numpy as np
from tamagotchi.main import main
import random
import time

# Setting up schema
schema = dj.schema("jq_fly_apparent_wind_vis_fb_noise_flip_wind")

@schema
class TrainingConfig(dj.Manual):
    definition = """
    training_config_hash: char(32)
    ---
    algo: varchar(64)
    lr: double
    eps: double
    alpha: double
    gamma: double
    use_gae: bool
    gae_lambda: double
    entropy_coef: double
    value_loss_coef: double
    max_grad_norm: double
    seed: int
    cuda_deterministic: bool
    num_processes: int
    num_steps: int
    ppo_epoch: int
    num_mini_batch: int
    clip_param: double
    log_interval: double
    save_interval: double
    no_cuda: bool
    use_proper_time_limits: bool
    recurrent_policy: bool
    use_linear_lr_decay: bool
    env_name: varchar(64)
    log_dir: varchar(64)
    save_dir: varchar(256)
    outsuffix: varchar(256)
    dynamic: bool
    eval_type: varchar(64)
    eval_episodes: int
    eval_interval = NULL: int
    weight_decay: double
    rnn_type: varchar(64)
    hidden_size: int
    betadist: bool
    stacking: int
    masking = NULL: varchar(64)
    stride: int
    dataset: blob
    num_env_steps: int
    qvar: blob
    birthx: double
    diff_max: blob
    diff_min: blob
    birthx_linear_tc_steps: int
    birthx_max: double
    dryrun: bool
    curriculum: bool
    turnx: double
    movex: double
    auto_movex: bool
    auto_reward: bool
    loc_algo: varchar(64)
    time_algo: varchar(64)
    env_dt: double
    walking: bool
    radiusx: double
    diffusion_min: double
    diffusion_max: double
    r_shaping: blob
    wind_rel: bool
    action_feedback: bool
    squash_action: bool
    flipping: bool
    odor_scaling: bool
    stray_max: double
    test_episodes: int
    viz_episodes: int
    model_fname: varchar(256)
    obs_noise: double
    act_noise: double
    cuda: bool
    apparent_wind: bool
    apparent_wind_allo: bool
    visual_feedback: bool
    if_vec_norm: int
    if_train_actor_std: bool
    rotate_by: bool
    """
    
    def get_new_seed(self):
        # return a random seed that is NOT already in the table
        while True:
            seed = random.randint(0, 32767) # keep tradition of $RANDOM in bash, where RAND~Uni(0, 32767)
            if seed not in self.fetch()['seed']:
                return seed
    

    def insert1(self, input_dict):
        # input_dict: dictionary with training config parameters
        dict_to_insert = input_dict.copy()
        # set novel seed if not provided
        if 'seed' not in dict_to_insert.keys():
            print("No seed provided. Generating a new one.")
            seed = self.get_new_seed()
            dict_to_insert['seed'] = seed
        
        # dummy outsuffix
        if 'outsuffix' not in dict_to_insert.keys():
            dict_to_insert['outsuffix'] = ''
        # Compute hash
        buffer = ""
        for key in self.heading.secondary_attributes:
            buffer += str(dict_to_insert[key])
        dict_to_insert['training_config_hash'] = hashlib.md5(buffer.encode()).hexdigest()
        
        # outsuffix of the form: seed_hash 
        if dict_to_insert['outsuffix'] == '':
            # if no outsuffix is provided, create one from seed and hash
            dict_to_insert['outsuffix'] = "_".join([str(dict_to_insert['seed']), dict_to_insert['training_config_hash']])
        # insert into table
        super().insert1(dict_to_insert)

    def json_insert(self, fjson):
        # TODO insert from json files. In case the table gets wiped from server
        return 
    
    
@schema
class TrainingResult(dj.Computed):
    definition = """
    -> TrainingConfig
    ---
    hours_elapsed: double
    seed: int
    """

    def make(self, key):
        # key is a dictionary containg the primary key of TrainingConfig [1 row]
        
        # start timer
        t0 = time.time()
        
        # Pull all the secondary attributes from the table into the dictionary
        args = (TrainingConfig & key).fetch1()

        # Pass it on your code
        main(args = args)

        # stop timer
        t1 = time.time()
        # calculate time taken in hours
        t = (t1-t0)/3600

        key['hours_elapsed'] = t
        key['seed'] = int(args['seed'])
        # EXAMPLE COMPLETE TRIANING
        self.insert1(key)
        
        
# @schema
# class EvalConfig(dj.Computed):
#     definition = """
#     -> TrainingConfig
#     ---
#     seed: int
#     algo: varchar(64)
#     dataset: varchar(64)
#     model_fname: varchar(256)
#     test_episodes: int
#     viz_episodes: int
#     fixed_eval: bool
#     test_sparsity: bool
#     device: varchar(64)
#     diffusionx: double
#     """

#     def make(self, key):
#         # key: hash with keys from TrainingResult head column - training_config_hash
        
#         # Pull all the secondary attributes from the table into the dictionary
#         args = (TrainingConfig & key).fetch1()
        
#         # Pass it on your code
#         keys = ['seed', 'algo', 'dataset', 'model_fname', 'fixed_eval', 'test_sparsity', 'device', 'diffusionx']
#         keys['viz_episodes'] = 10
#         for k in keys:
#             key[k] = args[k]
#         self.insert1(key)
                


# @schema
# class EvalResult(dj.Computed):
#     definition = """
#     -> TrainingResult
#     ---
#     hours_elapsed: double
#     """

#     def make(self, key):
#         # key: hash with keys from TrainingResult head column - training_config_hash
        
#         # start timer
#         t0 = time.time()
        
#         # Pull all the secondary attributes from the table into the dictionary
#         args = (TrainingConfig & key).fetch1()
#         # Pass it on your code
#         print(key['training_config_hash'])

#         # stop timer
#         t1 = time.time()
#         # calculate time taken in hours
#         t = (t1-t0)/3600

#         # key['hours_elapsed'] = t
#         # # EXAMPLE COMPLETE TRIANING
#         # self.insert1(key)
        