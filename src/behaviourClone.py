#@title imports
# As usual, a bit of setup
import os
import shutil
import time
import numpy as np
import torch
import pickle

import utilities.pytorch_util as ptu

# from deeprl.infrastructure.rl_trainer import RL_Trainer
from utilities.trainers import BC_Trainer
# from deeprl.agents.bc_agent import BCAgent
# from deeprl.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from policies.MLP_policy import MLPPolicySL

# %load_ext autoreload
# %autoreload 2

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def remove_folder(path):
    # check if folder exists
    if os.path.exists(path): 
        print("Clearing old results at {}".format(path))
        # remove if exists
        shutil.rmtree(path)
    else:
        print("Folder {} does not exist yet. No old results to delete".format(path))

bc_base_args_dict = dict(
    expert_policy_file = 'deeprl/policies/experts/Hopper.pkl', #@param
    expert_data = 'deeprl/expert_data/expert_data_Hopper-v2.pkl', #@param
    env_name = 'Hopper-v2', #@param ['Ant-v2', 'Humanoid-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Hopper-v2']
    exp_name = 'test_bc', #@param
    do_dagger = True, #@param {type: "boolean"}
    ep_len = 1000, #@param {type: "integer"}
    save_params = False, #@param {type: "boolean"}

    # Training
    num_agent_train_steps_per_iter = 3000, #@param {type: "integer"})
    n_iter = 1, #@param {type: "integer"})

    # batches & buffers
    batch_size = 10000, #@param {type: "integer"})
    eval_batch_size = 1000, #@param {type: "integer"}
    train_batch_size = 100, #@param {type: "integer"}
    max_replay_buffer_size = 1000000, #@param {type: "integer"}

    #@markdown network
    n_layers = 2, #@param {type: "integer"}
    size = 64, #@param {type: "integer"}
    learning_rate = 5e-3, #@param {type: "number"}

    #@markdown logging
    video_log_freq = -1, #@param {type: "integer"}
    scalar_log_freq = 1, #@param {type: "integer"}

    #@markdown gpu & run-time settings
    no_gpu = False, #@param {type: "boolean"}
    which_gpu = 0, #@param {type: "integer"}
    seed = 2, #@param {type: "integer"}
    logdir = 'test',
)

bc_args = dict(bc_base_args_dict)

env_str = 'HalfCheetah'
bc_args['expert_policy_file'] = 'deeprl/policies/experts/{}.pkl'.format(env_str)
bc_args['expert_data'] = 'deeprl/expert_data/expert_data_{}-v2.pkl'.format(env_str)
bc_args['env_name'] = '{}-v2'.format(env_str)

# Delete all previous logs
remove_folder('logs/behavior_cloning/{}'.format(env_str))

for seed in range(1):
    print("Running behavior cloning experiment with seed", seed)
    bc_args['seed'] = seed
    bc_args['logdir'] = 'logs/behavior_cloning/{}/seed{}'.format(env_str, seed)
    bctrainer = BC_Trainer(bc_args)
    bctrainer.run_training_loop()

with open("bc_policy.pkl", "wb") as file:
    pickle.dump(bctrainer.rl_trainer.agent.actor, file)