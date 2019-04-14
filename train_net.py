import torch, os, argparse
from torch import nn
from torch.backends import cudnn
from config.default import _C as cfg
from CheckPointer import CheckPointer
# from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer as CheckPointer
import torch.distributed as dist
from model import build_model
from data_loader import build_data_loader
from solver import build_loss, build_scheduler, build_optimizer
from utils import do_test, do_train, setup_logger, synchronize, get_rank


def train_net(cfg, logger, is_distributed=False, local_rank=0):
    model = build_model(cfg)

    p_sum = 0
    for p in model.named_parameters():
        logger.info('%s, %s, %s' % (p[0], p[1].shape, p[1].requires_grad))
        if p[1].requires_grad:
            p_sum += p[1].numel()

    logger.info('model learnabel parameters: %d\n' % p_sum)

    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[local_rank],
                                                    output_device=local_rank,
                                                    broadcast_buffers=False)

    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_scheduler(cfg, optimizer)

    arguments = {}
    arguments["epoch"] = 0

    checkpointer = CheckPointer(cfg=cfg,
                                logger=logger,
                                model=model,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                save_dir=cfg.OUTPUT_DIR,
                                #is_distributed=is_distributed,
                                save_to_disk=get_rank()==0
                                )

    extra_checkpoint_data = checkpointer.load()

    arguments.update(extra_checkpoint_data)

    criterion = build_loss(cfg)

    train_loader = build_data_loader(cfg, True, is_distributed=is_distributed)
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


def test_net(cfg, logger, model, is_distributed=False):
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    test_loader = build_data_loader(cfg, is_train=False, is_distributed=is_distributed)
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
    parser.add_argument(
        '--local_rank',
        default=0,
        help='set GPU id',
        type=int
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    if not os.path.isdir(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)


    is_distributed = (num_gpus > 1)
    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()    

    if cfg.MODEL.DEVICE is 'cuda':
        cudnn.benchmark = True
    

    logger = setup_logger('resnet', cfg.OUTPUT_DIR, distributed_rank=get_rank())
    logger.info('run with config: \n{}'.format(cfg))
    logger.info('use %d GPUs' % num_gpus)
    logger.info('local_ranki is: %d' % args.local_rank)

    model = train_net(cfg, logger, is_distributed=is_distributed, local_rank=args.local_rank)
    test_net(cfg, logger, model, is_distributed=is_distributed)

    # begin test


