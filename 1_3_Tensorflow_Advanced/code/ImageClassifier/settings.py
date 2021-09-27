import argparse
import logging


class Settings:
    def __init__(self):
        _args = self._get_arguments()

        self.num_epoch = _args.num_epoch
        self.ld_epoch = _args.ld_epoch
        self.batch_size = _args.batch_size
        self.eval_size = _args.eval_size
        self.learning_rate = _args.learning_rate
        self.net_type = _args.net_type
        self.ckpt_dir = _args.ckpt_dir
        self.tb_log_dir = _args.tb_log_dir
        self.dataset = _args.dataset

    def _get_arguments(self):
        parser = argparse.ArgumentParser(
            'Implementation for MNIST handwritten digits 2020')
        parser.add_argument('--num_epoch',
                            type=int,
                            default=400,
                            help='Parameter for total number of epoch',
                            required=False)
        parser.add_argument('--ld_epoch',
                            type=int,
                            default=300,
                            help='Learning Rate Linear Decay Start Epoch',
                            required=False)
        parser.add_argument('--batch_size',
                            type=int,
                            default=100,
                            help='parameter for batch size',
                            required=False)
        parser.add_argument('--eval_size',
                            type=int,
                            default=1000,
                            help='parameter for batch size',
                            required=False)
        parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.0005,  # VGG16
            # default=0.0001, # ResNet34
            help='Parameter for learning rate',
            required=False)
        parser.add_argument(
            '--net_type',
            type=str,
            # default='Dense',
            # default='Conv',
            # default='VGG16',
            default='Resnet34',
            # default='Resnet50',
            help='Parameter for Network Selection',
            required=False)
        parser.add_argument(
            '--ckpt_dir',
            type=str,
            default='./ckpt',
            help='The directory where the checkpoint files are located',
            required=False)
        parser.add_argument(
            '--tb_log_dir',
            type=str,
            default='./tb_logs',
            help='The directory where the Training logs are located',
            required=False)
        parser.add_argument(
            '--dataset',
            type=str,
            default='cifar10',
            # default='mnist',
            help='The dataset to use; mnist / cifar10',
            required=False)
        return parser.parse_args()

    def get_lr(self, epoch):
        decay_width = self.num_epoch - self.ld_epoch
        if epoch < self.ld_epoch:
            lr = self.learning_rate
        else:
            lr = 0.5 * self.learning_rate * (
                1. - float(epoch - self.ld_epoch) / float(decay_width))
            print('Linear Decay - Learning Rate is {:1.8f}'.format(lr))
        return lr
