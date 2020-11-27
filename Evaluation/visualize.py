import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class Visualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.num_units = model.num_units
        self.width = model.width
        self.height = model.height

    def visualize(self):
        imgs = torch.rand(size=(self.num_units, self.width, self.height)).to(self.device)
        imgs.requires_grad_(True)
        optim = torch.optim.SGD(params=[imgs], momentum=0, lr=0.2, weight_decay=0.1)

        for i in range(100):
            y = self.model.to_encode(imgs)
            target = -y.trace()
            optim.zero_grad()
            target.backward()
            optim.step()

        self.imgs = imgs.detach().cpu().numpy()

    def show(self):
        fig, a = plt.subplots(np.math.ceil(self.num_units / 8), 8, figsize=(4, np.math.ceil(self.num_units / 16)))
        for i, img in enumerate(self.imgs):
            axis: Axes = a[i // 8][i % 8]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(img, cmap='seismic')
        plt.show()
