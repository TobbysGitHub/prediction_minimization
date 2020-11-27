import numpy as np
import torch
from torch import nn

from Model.unit_linear import UnitLinear


class Predictor(nn.Module):
    def __init__(self, num_units, dim_hidden):
        super().__init__()
        self.num_units = num_units
        self.dim_hidden = dim_hidden

        self.model = nn.Sequential(
            UnitLinear(num_units, num_units, dim_hidden),
            nn.LeakyReLU(),
            UnitLinear(num_units, dim_hidden, 1),
        )

        self.temperature = nn.Parameter(torch.ones(self.num_units))

    def forward(self, x):
        x = x.detach()
        key = x  # s_b * n_u

        x = x.view(x.shape[0], 1, self.num_units) \
            .expand(-1, self.num_units, -1)  # s_b * n_u * n_u
        # self mask
        x = self_mask(x)
        # project to units
        query = self.model(x).squeeze(-1)  # s_b * n_u

        weights = -torch.pow(key - query.unsqueeze(1), 2)  # s_b(q) * s_b(k) * num_units
        weights = weights * self.temperature

        weights = torch.softmax(weights, dim=1)

        return weights


def self_mask(x: torch.Tensor):
    mask = torch.eye(x.shape[1], device=x.device) == 1
    return x.masked_fill(mask, 0)
