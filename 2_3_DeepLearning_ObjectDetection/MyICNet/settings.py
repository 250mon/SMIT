import argparse
import logging


class Settings:
    def __init__(self):
        _args = self._get_arguments()
        self.batch_size = 20
        self.learning_rate = 0.01
        self.num_epoch = _args.num_epoch
        self.ld_epoch = _args.ld_epoch
        self.ckpt_dir = _args.ckpt_dir
        self.tb_log_dir = _args.tb_log_dir
        self.res_dir = _args.res_dir
        self.random = _args.random
        self.random_type = _args.random_type

    def _get_arguments(self):
        parser=argparse.ArgumentParser('Implementation for PyNet handwritten digits 2021')
        parser.add_argument('--num_epoch',
                            type=int,
                            default=250,
                            help='Total number of training epochs',
                            required=False)
        parser.add_argument('--ld_epoch',
                            type=int, 
                            default=250,
                            help='Learning rate linear decay start epoch',
                            required=False)
        parser.add_argument('--ckpt_dir', 
                            type=str, 
                            default='./ckpt',
                            help='The directory where the checkpoint files are located', 
                            required=False)
        parser.add_argument('--tb_log_dir', 
                            type=str, 
                            default='./tb_logs',
                            help='The directory where the training logs are located',
                            required=False)
        parser.add_argument('--res_dir', 
                            type=str, 
                            default='./res',
                            help='The directory where the training results are located',
                            required=False)
        parser.add_argument('--random',
                            action='store_true',
                            help='Run the pretrained network on a random image',
                            required=False)
        parser.add_argument('--random_type',
                            type=str,
                            default='train',
                            choices=['train', 'test'],
                            help='Specify where a random image is taken from',
                            required=False)

        return parser.parse_args()


    def get_lr(self, epoch):
        decay_width = self.num_epoch - self.ld_epoch
        if epoch < self.ld_epoch:
            lr = self.learning_rate
        else:
            lr = 0.5 * self.learning_rate * (1. - float(epoch - self.ld_epoch)/float(decay_width))
            print('Linear Decay - Learning Rate is {:1.8f}'.format(lr))
        return lr

