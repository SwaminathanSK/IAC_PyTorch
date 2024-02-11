import torch
from torch import nn, distributions

from models.model import Model
from utilities.utils import t
from utilities import pytorch_util as ptu


class Actor(Model):

    name = "actor"

    def __init__(self):
        super(Actor, self).__init__()

        self.fc_base = None
        self.fc_mean = None
        self.fc_logsd = None
        self.optimiser = None

    def set_params(self, hyper_ps):
        hidden_size = hyper_ps['a_hidden_size']
        hidden_layers = hyper_ps['a_hidden_layers']
        action_dim = hyper_ps['action_dim']

        fcs = [nn.Linear(hyper_ps['state_dim'], hidden_size), nn.ReLU()]
        for _ in range(hidden_layers):
            fcs.append(nn.Linear(hidden_size, hidden_size))
            fcs.append(nn.ReLU())

        self.fc_base = nn.Sequential(*fcs)
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        # self.fc_logsd = nn.Linear(hidden_size, action_dim)
        self.logstd = nn.Parameter(
                torch.zeros(action_dim, dtype=torch.float32, device=ptu.device)
            )

        self.optimiser = torch.optim.SGD(
            lr=hyper_ps['a_learning_rate'],
            momentum=hyper_ps['a_momentum'],
            params=self.parameters()
        )

    def forward(self, state):
        x = self.fc_base(state)
        mean = self.fc_mean(x)
        # logsd = self.fc_logsd(x)
        logsd = torch.clamp(self.logstd, -10, 2) 
        # sd = logsd.exp()

        scale_tril = torch.diag(torch.exp(logsd))
        batch_dim = mean.shape[0]
        batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)

        normal = distributions.MultivariateNormal(loc=mean, scale_tril=batch_scale_tril)
        action = normal.sample()
        print(action)

        return normal, action

    def backward(self, loss):
        self.optimiser.zero_grad()
        loss.backward(t([1. for _ in range(loss.size()[0])]))
        self.optimiser.step()
