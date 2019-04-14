import torch
from torchvision.transforms import transforms
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import distributed


def make_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        ## distributed.DistributedSampler is shuffle by default
        return distributed.DistributedSampler(dataset)

    if shuffle:
        return torch.utils.data.sampler.RandomSampler(dataset)
    else:
        return torch.utils.data.sampler.SequentialSampler(dataset)


def build_data_loader(cfg, is_train, is_distributed=False):
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

    sampler = make_data_sampler(dataset, shuffle=is_train, is_distributed=is_distributed)
    loader = DataLoader(dataset, batch_size,
                        sampler=sampler,
                        num_workers=cfg.DATALOADER.NUM_WORKERS)
    return loader