import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tamagotchi.a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, trainable_std_dev=False):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        if trainable_std_dev:
            # Separate initialization for logstd with negative bias
            self.logstd = nn.Linear(num_inputs, num_outputs)
            nn.init.orthogonal_(self.logstd.weight)
            nn.init.constant_(self.logstd.bias, -0.5)  # Start with std ≈ 0.6
        else:
            self.logstd = AddBias(torch.zeros(num_outputs))
        self.trainable_std_dev = trainable_std_dev

    def forward(self, x):
        action_mean = self.fc_mean(x)

        if self.trainable_std_dev:
            action_logstd = self.logstd(x)
        else:
            #  An ugly hack for my KFAC implementation.
            zeros = torch.zeros(action_mean.size())
            if x.is_cuda:
                zeros = zeros.cuda()
            action_logstd = self.logstd(zeros)
        # CLAMP BEFORE EXP TO AVOID EXPLOSIVE VALUES
        action_logstd = torch.clamp(action_logstd, min=-20, max=2)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


# Using slightly modified version of implementation from: 
# https://github.com/dhruvms/HighwayTraffic/blob/4fb5a1d2cba72b182198a61e81ab3022b95d4d4b/python/external/pytorch_baselines/algo/distributions.py
class FixedBeta(torch.distributions.Beta):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return (self.concentration1 - 1) / \
          (self.concentration1 + self.concentration0 - 2)

class BetaCustom(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(BetaCustom, self).__init__()

        self.alpha = nn.Sequential(
                        nn.Linear(num_inputs, num_outputs),
                        nn.Softplus())
        self.beta = nn.Sequential(
                        nn.Linear(num_inputs, num_outputs),
                        nn.Softplus())

    def forward(self, x):
        alpha = self.alpha(x)
        ones_a = torch.ones(1)
        alpha_plus =  alpha + ones_a.expand(alpha.size())

        beta = self.beta(x)
        ones_b = torch.ones(1)
        beta_plus =  beta + ones_b.expand(beta.size())

        return FixedBeta(alpha_plus, beta_plus)