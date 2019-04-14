import torch
from torch import nn
from torchvision.models import resnet
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
