import os, logging, time, torch, sys
from collections import deque
import torch.distributed as dist
import numpy as np

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

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


def do_train(cfg, logger, model, device, train_loader, optimizer, lr_scheduler, criterion, checkpointer, arguments):

    epoch_num = cfg.TRAIN.EPOCH
    
    start_epoch = arguments['epoch']
    logger.info('take %d iter steps: ' % ((epoch_num-start_epoch) * len(train_loader)))
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
    logger.info('begin testing')
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
