import gym
import d4rl

from agent import AWRAgent
from models.actor import Actor
# from src.trash.critic import Critic
from critics.bootstrapped_continuous_critic import BootstrappedContinuousCritic
from utilities.debug import DebugType
from utilities.training import Training


# setting the hyper-parameters
hyper_ps = {
    'replay_size': 50000,
    'max_epochs': 150,
    'sample_mod': 10,
    'beta': 0.25, # IAC
    'max_advantage_weight': 200., # IAC
    'min_log_pi': -50.,
    'discount_factor': .99, # IAC
    'alpha': 0.95, 
    'c_hidden_size': 256, # IAC
    'c_hidden_layers': 2, # IAC
    'a_hidden_size': 256, # IAC
    'a_hidden_layers': 2, # IAC
    'c_learning_rate': 3e-4, # IAC
    'c_momentum': .9,
    'a_learning_rate': 3e-4, # IAC
    'a_momentum': .9, 
    'critic_threshold': 17.5,
    'critic_suffices_required': 1,
    'critic_steps_start': 200,
    'critic_steps_end': 200,
    'actor_steps_start': 1000,
    'actor_steps_end': 1000, 
    'batch_size': 256, # IAC
    'seed': 0, # IAC
    'replay_fill_threshold': 1.,
    'random_exploration': True,
    'test_iterations': 30,
    'validation_epoch_mod': 3,
}

# configuring the environment
environment = gym.make("halfcheetah-expert-v2")
# environment._max_episode_steps = 600

# behavioural = AWRAgent
# actor = Actor()
# critic = Critic()

# setting up the training components
agent = AWRAgent
actor = Actor()
critic = BootstrappedContinuousCritic({'target_update_rate' : 0.005}) # IAC


# training and testing
Training.train(
    (actor, critic),
    agent,
    environment,
    hyper_ps,
    save=True,
    debug_type=DebugType.NONE
)
