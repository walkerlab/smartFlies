import glob
import os

import torch
import torch.nn as nn

# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None
 
    
# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# Wind obsver util: wind negative log likelihood loss
def wind_nll_stats(mu, logvar, target):
    """
    mu, logvar, target: [B, 2]
    Returns:
      mean_nll: scalar
      per_sample_nll: [B]
    """
    var = logvar.exp()

    per_sample = 0.5 * ((((target - mu) ** 2) / var) + logvar).sum(-1)  # [B]； (target - mu) ** 2) / var) = MSE / var (inflates error when uncertain); + logvar (penalize uncertainty)
    mean_nll = per_sample.mean()
    return mean_nll, per_sample
