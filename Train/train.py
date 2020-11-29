import torch
from tqdm import tqdm

CTR = False


def cal_loss(y1, y2, w):
    """
    :param y1:  s_b * n_u
    :param y2:  s_b * n_u
    :param w:   s_b(q) * s_b * num_units
    """

    l12 = torch.exp(-torch.abs(y1 - y2))  # s_b * n_u
    l1neg = torch.sum(w * torch.exp(-torch.abs(y1.unsqueeze(1) - y1)), dim=1)  # s_b * n_u
    l2neg = torch.sum(w * torch.exp(-torch.abs(y2.unsqueeze(1) - y1)), dim=1)
    loss = -torch.log(l12 / l1neg) - torch.log(l12 / l2neg)
    loss = loss.mean()

    return loss


def cal_loss(y1, y2, w, m):
    l_1_2 = torch.exp(-torch.abs(y1 - y2).clamp_max(5))  # s_b * n_u
    l_1_neg = torch.sum(w * torch.exp(-torch.abs(y1.unsqueeze(1) - y1)), dim=1)  # s_b * n_u
    l_2_neg = torch.sum(w * torch.exp(-torch.abs(y2.unsqueeze(1) - y1)), dim=1)

    loss = -torch.log(l_1_2 / l_1_neg) - torch.log(l_1_2 / l_2_neg)
    loss = loss.masked_fill(~m, 0)
    loss = loss.mean()

    return loss


def ctr_loss(y1, y2):
    """
    :param y1:  s_b * n_u
    :param y2:  s_b * n_u
    """

    l12 = torch.exp(-torch.abs(y1 - y2))  # s_b * n_u
    l1neg = torch.mean(torch.exp(-torch.abs(y1.unsqueeze(1) - y1)), dim=1)  # s_b * n_u
    l2neg = torch.mean(torch.exp(-torch.abs(y2.unsqueeze(1) - y1)), dim=1)
    loss = -torch.log(l12 / l1neg) - torch.log(l12 / l2neg)
    loss = loss.mean()

    return loss


def train_epoch(model, data_loader, optimizer, ):
    for batch in data_loader:
        y1, y2, w, m = model(batch[0])
        if not CTR:
            loss = cal_loss(y1, y2, w, m)
        else:
            loss = ctr_loss(y1, y2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    pass


def train(model, data_loader, optimizer, epochs):
    for epoch in tqdm(range(epochs)):
        train_epoch(model, data_loader, optimizer, )
