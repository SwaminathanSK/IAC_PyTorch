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
    num_agent_train_steps_per_iter = 10000, #@param {type: "integer"})
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

### Basic test for correctness of loss and gradients
torch.manual_seed(0)
ac_dim = 6
ob_dim = 17
batch_size = 5

policy = MLPPolicySL(
            ac_dim=ac_dim,
            ob_dim=ob_dim,
            n_layers=1,
            size=2,
            learning_rate=0.25)

np.random.seed(0)
obs = np.random.normal(size=(batch_size, ob_dim))
acts = np.random.normal(size=(batch_size, ac_dim))

first_weight_before = np.array(ptu.to_numpy(next(policy.mean_net.parameters())))
print("Weight before update", first_weight_before)

for i in range(5):
    loss = policy.update(obs, acts)['Training Loss']

print(loss)
expected_loss = 2.628419
loss_error = rel_error(loss, expected_loss)
print("Loss Error", loss_error, "should be on the order of 1e-6 or lower")

first_weight_after = ptu.to_numpy(next(policy.mean_net.parameters()))
print('Weight after update', first_weight_after)

weight_change = first_weight_after - first_weight_before
print("Change in weights", weight_change)

expected_change = np.array([[ 0.04385546, -0.4614172,  -1.0613215 ],
                            [ 0.20986436, -1.2060736,  -1.0026767 ]])
updated_weight_error = rel_error(weight_change, expected_change)
print("Weight Update Error", updated_weight_error, "should be on the order of 1e-6 or lower")

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
    pickle.dump(policy, file)