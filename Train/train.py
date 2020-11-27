import torch
from tqdm import tqdm


def cal_loss(y1, y2, w):
    """
    :param y1:  s_b * n_u
    :param y2:  s_b * n_u
    :param w:   s_b(q) * s_b * num_units
    :param mask: s_b * num_units
    """

    l12 = torch.exp(-torch.abs(y1 - y2))  # s_b * n_u
    l1neg = torch.sum(w * torch.exp(-torch.abs(y1.unsqueeze(1) - y1)), dim=1)  # s_b * n_u
    l2neg = torch.sum(w * torch.exp(-torch.abs(y2.unsqueeze(1) - y1)), dim=1)
    loss = -torch.log(l12 / l1neg) - torch.log(l12 / l2neg)
    loss = loss.mean()

    return loss


def train_epoch(model, data_loader, optimizer, ):
    for batch in data_loader:
        y1, y2, w = model(batch[0])
        loss = cal_loss(y1, y2, w)
        loss.backward()
        optimizer.step()
    pass


def train(model, data_loader, optimizer, epochs):
    for epoch in tqdm(range(epochs)):
        train_epoch(model, data_loader, optimizer, )
