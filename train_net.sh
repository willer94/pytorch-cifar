export NGPUS=1
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_net.py --config-file "config/cifar100_config.yaml"
