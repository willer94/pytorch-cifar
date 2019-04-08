"""
more details can be found in maskrcnn_benchmark (https://github.com/facebookresearch/maskrcnn-benchmark)
"""
import os
import torch

class CheckPointer(object):
    def __init__(self, cfg, logger, model, optimizer=None, lr_scheduler=None, save_dir=''):
        self.cfg = cfg.clone()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_dir = save_dir
        self.logger = logger

    def save(self, name, **kwargs):
        data = {}
        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data['optimizer'] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            data['lr_scheduler'] = self.lr_scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, name+'.pth')
        torch.save(data, save_file)
        self.logger.info('save model to %s\n' % save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            f = self.get_checkpoint_file()
        if f is None:
            print('no checkpoint file was found')
            return {}

        checkpoint = torch.load(f, map_location=torch.device("cpu"))
        self.logger.info('load model from %s\n' % f)
        self.model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
        if 'lr_scheduler' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint.pop('lr_scheduler'))
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)
