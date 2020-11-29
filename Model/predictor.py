import numpy as np
import torch
from torch import nn

from Model.unit_linear import UnitLinear


class Predictor(nn.Module):
    def __init__(self, num_units, dim_hidden):
        super().__init__()
        self.num_units = num_units
        self.dim_hidden = dim_hidden
        self.num_groups = 1

        self.model1 = nn.Sequential(
            UnitLinear(num_units, num_units, dim_hidden),
            nn.LeakyReLU(),
            UnitLinear(num_units, dim_hidden, 1),
        )
        self.model2 = nn.Sequential(
            nn.Linear(self.num_units, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.num_units)
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
        query = self.model1(x).squeeze(-1)  # s_b * n_u

        weights = -torch.pow(key - query.unsqueeze(1), 2)  # s_b(q) * s_b(k) * num_units
        weights = weights * self.temperature
        weights = eye_mask(weights)

        weights = torch.softmax(weights, dim=1)

        return weights, None

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.detach()
        x = x.view(batch_size, self.num_units)
        # project to units
        key = x

        _, mask = random_mask(x)  # s_b * num_units,  s_b * num_units
        # query = self.model2(x)

        x = x.view(x.shape[0], 1, self.num_units) \
            .expand(-1, self.num_units, -1)  # s_b * n_u * n_u
        # self mask
        x = self_mask(x)
        # project to units
        query = self.model1(x).squeeze(-1)  # s_b * n_u

        weights = -torch.abs(key - query.unsqueeze(1))  # s_b(q) * s_b(k) * num_units
        weights = weights * self.temperature
        weights = eye_mask(weights)

        weights = weights.view(batch_size, batch_size // self.num_groups, self.num_groups, self.num_units)
        weights = torch.softmax(weights, dim=1)
        weights = weights.view(batch_size, batch_size, self.num_units) / 8

        return weights, mask


def self_mask(x: torch.Tensor):
    mask = torch.eye(x.shape[1], device=x.device) == 1
    return x.masked_fill(mask, 0)


def eye_mask(x: torch.Tensor):
    mask = torch.eye(x.shape[1], device=x.device) == 1
    return x.masked_fill(mask.unsqueeze(-1), -np.inf)


def random_mask(x: torch.Tensor):
    batch_size, num_unit = x.shape
    mask = torch.rand_like(x) < 0.2
    mask_num = torch.sum(mask, dim=1).unsqueeze(-1)
    x = x.masked_fill(mask, 0)
    x = x * (torch.Tensor([num_unit]).to(x.device).float() / mask_num.float())

    return x, mask
