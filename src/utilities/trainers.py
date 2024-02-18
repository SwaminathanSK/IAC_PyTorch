from .rl_trainer import RL_Trainer # To change this
from agents.bc_agent import BCAgent
# from deeprl.agents.pg_agent import PGAgent
# from deeprl.agents.dqn_agent import DQNAgent
# from deeprl.agents.ac_agent import ACAgent 
from policies.loaded_gaussian_policy import LoadedGaussianPolicy

# from deeprl.infrastructure.dqn_utils import get_env_kwargs

class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        self.params['agent_class'] = BCAgent ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        # print('Loading expert policy from...', self.params['expert_policy_file'])
        # self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        # print('Done restoring expert policy...')

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
        )