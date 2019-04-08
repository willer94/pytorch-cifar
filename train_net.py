import torch, os, time, argparse, sys, logging
from torch import nn, optim
from torchvision.models import resnet
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.backends import cudnn
from config.default import _C as cfg
from CheckPointer import CheckPointer
from collections import deque
import numpy as np

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def freeze_model(model, freeze_at):
    if freeze_at < 0:
        return
    for stage_idx in range(0, freeze_at):
        if stage_idx == 0:
            m = nn.Sequential(getattr(model, 'conv1'), getattr(model, 'bn1'))
        else:
            m = getattr(model, "layer" + str(stage_idx))
        for p in m.parameters():
            p.requires_grad = False

def build_model(cfg):
    model = resnet.resnet50(pretrained=True)
    freeze_at = cfg.MODEL.FREEZE_AT
    freeze_model(model, freeze_at)

    class_num = cfg.MODEL.CLASSES
    if class_num != 1000:
        infeatures = model.fc.in_features
        model.fc = nn.Linear(in_features=infeatures, out_features=class_num, bias=True)
    return model

def build_data_loader(cfg, is_train):
    # prepare dataset and dataloader
    DATASET = cfg.INPUT.DATASET

    data_path = cfg.INPUT.DATASET_ROOT

    mean, std = cfg.INPUT.MEAN, cfg.INPUT.STD
    if is_train:
        data_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize(224),
             # transforms.ColorJitter(),
             transforms.ToTensor(),  # torchvision的ToTensor 除以了255，归一化到[0, 1]
             transforms.Normalize(mean=mean, std=std),
             ]
        )
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        data_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize(224),
             # transforms.ColorJitter(),
             transforms.ToTensor(),  # torchvision的ToTensor 除以了255，归一化到[0, 1]
             transforms.Normalize(mean=mean, std=std),
             ]
        )
        batch_size = cfg.TEST.BATCH_SIZE
    if DATASET == 'cifar10':
        dataset = CIFAR10(root=data_path, train=is_train, transform=data_transforms)
    elif DATASET == 'cifar100':
        dataset = CIFAR100(root=data_path, train=is_train, transform=data_transforms)
    else:
        print('Not support dataset: ', DATASET)
        return {}
    loader = DataLoader(dataset, batch_size, shuffle=is_train)
    return loader


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

def do_train(cfg, logger, model, device, train_loader, optimizer, lr_scheduler, criterion, checkpointer, arguments):

    epoch_num = cfg.TRAIN.EPOCH
    logger.info('take %d iter steps: ' % (epoch_num * len(train_loader)))
    start_epoch = arguments['epoch']

    time_begin = time.time()
    for epoch in range(start_epoch, epoch_num):
        arguments['epoch'] = epoch+1
        model.train()
        lr_scheduler.step(epoch=epoch)

        max_len = cfg.TRAIN.MAX_LEN
        train_loss = deque(maxlen=max_len)
        correct = deque(maxlen=max_len)
        total = deque(maxlen=max_len)
        correct_all, total_all = 0, 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predict = outputs.max(1)

            train_loss.append(loss.item())
            correct_current = predict.eq(labels).sum().item()
            correct.append(correct_current)
            total.append(labels.shape[0])
            correct_all += correct_current
            total_all += labels.shape[0]
            logger.info('epoch: %3d, batch: %4d, loss: %.3f, Acc: %.3f%%, Acc all: %.3f%% (%d/%d)' %
                  (epoch, batch_idx,
                   np.asarray(train_loss).sum() / len(train_loss),
                   100. * np.asarray(correct).sum() / np.asarray(total).sum(),
                   100. * (correct_all / total_all), correct_all, total_all))

        if (epoch+1) % cfg.TRAIN.CHECK_PERIOD == 0:
            checkpointer.save("model_{:07d}".format(epoch), **arguments)
        if (epoch+1) == epoch_num:
            checkpointer.save("model_final", **arguments)

    time_cost = time.time() - time_begin
    logger.info('freeze at %d, batch_size: %d, time used: %.3f s' %
                (cfg.MODEL.FREEZE_AT, cfg.TRAIN.BATCH_SIZE, time_cost))
    return model


def do_test(cfg, logger, model, device, test_loader, criterion):
    torch.cuda.empty_cache()
    print('begin testing')
    test_loss, correct, total = 0, 0, 0
    test_len = 0
    with torch.no_grad():
        for _, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predict = outputs.max(1)
            total += labels.shape[0]
            correct += predict.eq(labels).sum().item()
            test_len += 1

        logger.info('test loss: %.3f, Acc: %.3f%% (%d/%d)' %
              (test_loss / test_len, 100. * (correct / total), correct, total))

def train_net(cfg, logger):
    model = build_model(cfg)

    p_sum = 0
    for p in model.named_parameters():
        print(p[0], p[1].shape, p[1].requires_grad)
        if p[1].requires_grad:
            p_sum += p[1].numel()

    logger.info('model learnabel parameters: %d\n' % p_sum)

    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)

    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_scheduler(cfg, optimizer)

    arguments = {}
    arguments["epoch"] = 0

    checkpointer = CheckPointer(cfg, logger, model, optimizer, lr_scheduler, save_dir=cfg.OUTPUT_DIR)
    extra_checkpoint_data = checkpointer.load()

    arguments.update(extra_checkpoint_data)

    criterion = build_loss(cfg)

    train_loader = build_data_loader(cfg, True)
    model = do_train(cfg=cfg,
                     model=model,
                     device=device,
                     train_loader=train_loader,
                     optimizer=optimizer,
                     lr_scheduler=lr_scheduler,
                     criterion=criterion,
                     checkpointer=checkpointer,
                     arguments=arguments,
                     logger=logger)
    return model


def test_net(cfg, logger, model):
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    test_loader = build_data_loader(cfg, is_train=False)
    criterion = build_loss(cfg)
    do_test(cfg=cfg,
            model=model,
            device=device,
            test_loader=test_loader,
            criterion=criterion,
            logger=logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 and CIFAR100 RESNET Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if cfg.MODEL.DEVICE is 'cuda':
        cudnn.benchmark = True
    if not os.path.isdir(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger('resnet', cfg.OUTPUT_DIR, 0)
    logger.info('run with config: \n{}'.format(cfg))

    model = train_net(cfg, logger)
    test_net(cfg, logger, model)

    # begin test


