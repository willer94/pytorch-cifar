from torch import nn, optim


def build_optimizer(cfg, model):
    lr = cfg.SOLVER.BASE_LR
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def build_scheduler(cfg, optimizer):
    milestones = cfg.SOLVER.MILESTONES
    gamma = cfg.SOLVER.GAMMA
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return lr_scheduler


def build_loss(cfg):
    return nn.CrossEntropyLoss()