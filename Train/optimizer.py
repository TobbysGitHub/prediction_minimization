import torch


class BiOptimizer(object):
    def __init__(self, model, lr=0.1):
        self.optimizers = [torch.optim.SGD(params=model.get_params0(),
                                           lr=lr, momentum=0.9, weight_decay=0.001),
                           torch.optim.SGD(params=model.get_params1(),
                                           lr=lr, momentum=0.9, weight_decay=0.001), ]
        super().__init__()

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step(self):
        optimizer = self.optimizers[0]
        optimizer.step()

        def flip_grad(o):
            for group in o.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data = - p.grad.data

        optimizer = self.optimizers[1]
        flip_grad(optimizer)
        optimizer.step()
