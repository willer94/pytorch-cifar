# pytorch-cifar
cifar10 and cifar100 classification using Pytorch Resnet.
for cifar10, run
```python
python train_net.py --config-file 'config/cifar10_config.yaml'
```

# result 
batch_size: 32, epoch: 20, base_lr: 0.01, gamma: 0.1, milestone: [4, 14]
cifar10: 96.56%
cifar100: 84.68%
