import copy

from .critic import Critic
import torch
from torch import nn
from torch import optim

from utilities import pytorch_util as ptu


class BootstrappedContinuousCritic(Critic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        # critic parameters
        self.target_update_rate = hparams['target_update_rate']
    
    def set_params(self, hyper_ps):
        super().set_params(hyper_ps)
        self.critic_network = self.fc
        self.critic_network.to(ptu.device)
        self.target_network = copy.deepcopy(self.critic_network)
        self.loss = nn.MSELoss()

    def forward_np(self, obs, acts):
        obs = ptu.from_numpy(obs)
        acts = ptu.from_numpy(acts)
        predictions = self(obs, acts)
        return ptu.to_numpy(predictions)

    def update_target_network_ema(self):
        with torch.no_grad():
            for p,q in zip(self.target_network.parameters(), self.critic_network.parameters()):
                new_val = (1-self.target_update_rate)*p + self.target_update_rate*q
                p.copy_(new_val)

    def update(self, obs, acts, next_obs, rewards, terminals, actor, target_value):
        """
            Update the parameters of the critic.

            arguments:
                obs: shape: (batch_size, ob_dim)
                acts: shape: (batch_size, acdim)
                next_obs: shape: (batch_size, ob_dim). The observation after taking one step forward
                reward_n: length: batch_size. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: batch_size. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        obs = ptu.from_numpy(obs)
        acts = ptu.from_numpy(acts)
        rewards = ptu.from_numpy(rewards)
        terminals = ptu.from_numpy(terminals)
        target_value = target_value

        q_pred = self.critic_network(obs)
        loss = self.loss(q_pred, target_value.detach())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # update target Q function with exponential moving average
        self.update_target_network_ema()

        return {
            'Critic Training Loss': ptu.to_numpy(loss),
            'Critic Mean': ptu.to_numpy(q_pred.mean()),
        }
