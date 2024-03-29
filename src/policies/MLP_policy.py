import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from utilities import pytorch_util as ptu
from .base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 entropy_weight=0.,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight # only used for actor critic
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        # action = self(observation)
        action_distribution = self(observation)
        action = action_distribution.sample()  # don't bother with rsample
        return ptu.to_numpy(action)

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function outputs a distribution object representing the policy output for the particular observations.
    # assumes first dimension of observations 
    def forward(self, observation: torch.FloatTensor):
        if self.discrete:
            logits = self.logits_na(observation)
            return torch.distributions.Categorical(logits=logits)
        else:
            batch_mean = self.mean_net(observation)
            logstd = torch.clamp(self.logstd, -10, 2) 
            scale_tril = torch.diag(torch.exp(logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            return distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        prob = self.forward(observations)
        loss = -(torch.sum(prob.log_prob(actions)))/np.shape(observations)[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }

class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, acs_na, adv_n=None, acs_labels_na=None,
                   qvals=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(acs_na)
        adv_n = ptu.from_numpy(adv_n)
        prob = self.forward(observations)
        loss = -(torch.sum(prob.log_prob(actions) * adv_n))/np.shape(observations)[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            # first standardize qvals to make this easier to train
            targets_n = (qvals - np.mean(qvals)) / (np.std(qvals) + 1e-8)
            targets_n = ptu.from_numpy(targets_n)
            baseline_loss = None
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        else:
            baseline_loss = None
        return {
            'Training Loss': ptu.to_numpy(loss),
            'Baseline Loss': ptu.to_numpy(baseline_loss) if baseline_loss else 0,
        }

    def run_baseline_prediction(self, observations):
        observations = ptu.from_numpy(observations)
        pred = self.baseline(observations)
        return ptu.to_numpy(pred.squeeze())

class MLPPolicyAC(MLPPolicy):
    def forward(self, observations: torch.FloatTensor):
        if self.discrete:
            return super(MLPPolicyAC).forward(observations)
        else:
            base_dist = super(MLPPolicyAC, self).forward(observations)
            # for AC methods, we need to ensure actions sampled from the env
            # are valid actions for the environment. 
            # Since the action spaces are bounded between [-1, 1], we apply
            # a tanh function to the Gaussian policy samples.
            return torch.distributions.transformed_distribution.TransformedDistribution(
                    base_dist, [torch.distributions.transforms.TanhTransform()])


    def update(self, observations, critic):
        observations = ptu.from_numpy(observations)
        loss = None
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            'Actor Training Loss': ptu.to_numpy(loss),
        }
