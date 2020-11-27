import torch
import torch.nn as nn

from . import Encoder
from . import Predictor


class Model(nn.Module):
    def __init__(self, num_units=64, width=16, height=16):
        super().__init__()
        self.num_units = num_units
        self.width = width
        self.height = height
        self.encoder = Encoder(num_units=num_units, dim_inputs=width * height, dim_hidden=32)
        self.predictor = Predictor(num_units=num_units, dim_hidden=4)

    @staticmethod
    def add_noise(x):
        noise = 0.1 * torch.randn_like(x)
        return x + noise

    def forward(self, x):
        x = x.view(-1, self.width * self.height)

        x1 = self.add_noise(x)
        y1 = self.encoder(x1)  # s_b * n_u
        w = self.predictor(y1)  # s_b(q) * s_b * num_units

        x2 = self.add_noise(x)
        y2 = self.encoder(x2)

        return y1, y2, w

    def to_encode(self, x):
        x = x.view(-1, self.width * self.height)
        y = self.encoder(x)

        return y

    def get_params0(self):
        return self.encoder.parameters()

    def get_params1(self):
        return self.predictor.parameters()
