import argparse
import logging


class Settings:
    def __init__(self):
        _args = self._get_arguments()
        self.num_epoch = 900
        self.batch_size = 16
        self.learning_rate = 0.001
        self.ld_epoch = _args.ld_epoch
        self.ckpt_dir = _args.ckpt_dir
        self.tb_log_dir = _args.tb_log_dir
        self.res_dir = _args.res_dir

    def _get_arguments(self):
        parser = argparse.ArgumentParser('Implementation for PyNet handwritten digits 2021')
        parser.add_argument('--ld_epoch',
                            type=int, 
                            default=300,
                            help='Learning Rate Linear Decay Start Epoch', 
                            required = False)
        parser.add_argument('--ckpt_dir', 
                            type=str, 
                            default='./ckpt',
                            help='The directory where the checkpoint files are located', 
                            required = False)
        parser.add_argument('--tb_log_dir', 
                            type=str, 
                            default='./tb_logs',
                            help='The directory where the Training logs are located', 
                            required = False)
        parser.add_argument('--res_dir', 
                            type=str, 
                            default='./res',
                            help='The directory where the Training results are located', 
                            required = False)

        return parser.parse_args()


    def get_lr(self, epoch):
        decay_width = self.num_epoch - self.ld_epoch
        if epoch < self.ld_epoch:
            lr =  self.learning_rate
        else:
            lr = 0.5 * self.learning_rate * (1. - float(epoch - self.ld_epoch)/float(decay_width))
            print('Linear Decay - Learning Rate is {:1.8f}'.format(lr))
        return lr

