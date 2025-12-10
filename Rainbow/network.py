import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import math
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'k'))
N_ATOMS = 51

# for QR-DQN
def quantile_huber_loss(curr, target, device):
    diff = target.unsqueeze(1) - curr.unsqueeze(2)
    huber = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)
    tau = torch.linspace(0.5 / N_ATOMS, 1 - 0.5 / N_ATOMS, N_ATOMS, device=device)
    loss = (torch.abs(tau.view(1, N_ATOMS, 1) - (diff.detach() < 0).float()) * huber).mean(dim = 2).sum(dim = 1)

    return loss

# Noisy Network
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # nn.Parameter -> It is learnable!
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)

        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

class QNetwork(Network):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layer(x)

class DuelingQNetwork(Network):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.feature = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        q = v + a - a.mean(dim = 1, keepdim = True)

        return q

class DuelingQRDQN(Network):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.feature = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, N_ATOMS)
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size * N_ATOMS)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature(x)
        v = self.value(x).view(batch_size, 1, N_ATOMS)
        a = self.advantage(x).view(batch_size, self.output_size, N_ATOMS)
        q = v + a - a.mean(dim=1, keepdim=True)

        return q

class RainbowDQN(Network):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.feature = nn.Sequential(
            NoisyLinear(input_size, 128),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, N_ATOMS)
        )

        self.advantage = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, output_size * N_ATOMS)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature(x)

        v = self.value(x).view(batch_size, 1, N_ATOMS)
        a = self.advantage(x).view(batch_size, self.output_size, N_ATOMS)

        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


