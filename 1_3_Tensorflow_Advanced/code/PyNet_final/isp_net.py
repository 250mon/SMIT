from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf

from settings import Settings
from network_params import NetParams
import util_datasets
import network_model
from util_ckpt import CheckPtHandler
from util_log import Loggers
import pdb

tf.disable_eager_execution()


class ISPNet:
    def __init__(self, name):
        self.settings = Settings()
        self.settings.tb_log_dir = self.settings.tb_log_dir + '_' + name
        self.settings.ckpt_dir = self.settings.ckpt_dir + '_' + name
        # network params created
        self.net_params = NetParams(self)
        # dataset
        self.dataset = util_datasets.PyNetReader(self)
        self.vgg19 = util_datasets.NeuralLoss()
        # tf graph related
        self.network = None
        self.ckpt_handler = None
        # log file and analysis file name will be the network type
        self.loggers = Loggers(name)
        # for local loggings
        self.logger_train = None
        self.logger_eval = None
        self.is_last_epoch = False

    def _build_graph(self):
        # creating network
        self.network = network_model.PyNet(self)
        # chpt handler
        self.ckpt_handler = CheckPtHandler(self)

    def train(self, epoch, level):
        curr_lr = self.settings.get_lr(epoch)
        num_of_batches = self.dataset.get_num_of_batches()
        # key for train_ops, loss_ops, outputs
        level_ = 'lv' + str(level)
        mean_cost = 0.

        for step in range(num_of_batches):
            if step % 100 == 0:
                print(f"Step {step}/{num_of_batches} running...")

            batch_images, batch_labels = self.dataset.next_batch()
            feed_dict = {
                self.network.t_batch_img: batch_images,
                self.network.t_batch_lab: batch_labels,
                self.net_params.ph_learning_rate: curr_lr,
            }

            _, loss, outputs = self.network.session.run(
                (self.network.train_ops[level_], self.network.t_loss_ops[level_],
                 self.network.t_outputs[level_]),
                feed_dict=feed_dict)

            # pdb.set_trace()
            mean_cost += loss

            # if it is the last step of the level, then show images
            if self.is_last_epoch and step == num_of_batches - 1:
                self.validate_imgs(outputs, batch_labels, f'e{epoch:05d}_s{step:04d}')
        # calculate mean of losses of minibatch
        mean_cost /= float(num_of_batches)
        return mean_cost

    # show or/and save the label/output images of training
    def validate_imgs(self, outputs, labels, title):
        v_min = outputs.min()
        v_max = outputs.max()
        outputs = ((outputs - v_min) / (v_max - v_min) * 255).astype(np.uint8)

        resized_labels = self.dataset.resize_label(labels, outputs)

        util_datasets.show_zurich(outputs,
                                  resized_labels,
                                  title,
                                  save=True)

    def evaluate(self, save_output=False):
        eval_num = self.dataset.get_eval_number()
        total_ssim = 0.
        total_psnr = 0.
        for step in range(eval_num):
            eval_images, eval_labels = self.dataset.next_eval_batch()
            feed_dict = {
                self.network.t_batch_img: eval_images,
                self.network.t_batch_lab: eval_labels,
            }
            # session run
            ssim, psnr, outputs = self.network.session.run(
                (self.network.t_loss_ops['eval_ssim'],
                 self.network.t_loss_ops['eval_psnr'],
                 self.network.t_outputs['lv1']),
                feed_dict=feed_dict)
            total_ssim += ssim
            total_psnr += psnr
            if save_output is True:
                self.validate_imgs(outputs, eval_labels, f'lv1_img{step:02d}')

        total_ssim /= eval_num
        total_psnr /= eval_num
        return total_ssim, total_psnr

    def run(self):
        print("---------------------------------------------------------")
        print("         Starting Zurich-Data Batch Processing Example")
        print("---------------------------------------------------------")

        # creating network
        self._build_graph()
        self.logger_train = self.loggers.create_logger(self.settings.net_type +
                                                       '_train')
        self.logger_eval = self.loggers.create_logger(self.settings.net_type +
                                                      '_eval')

        # num_epochs = [6, 6, 25, 25, 45, 120]
        # num_epochs = [2, 2, 5, 5, 7, 10]
        num_epochs = [2, 2, 2, 2, 2, 2]
        # num_epochs = [0, 0, 0, 0, 0, 0]

        # ckpt_epoch becomes the cum start epoch
        ckpt_epoch = self.ckpt_handler.restore_ckpt()
        # start_idx = 0
        start_level = 5
        start_epoch = ckpt_epoch
        next_lv_beg_idx = 0

        # cumulative sum of epochs in reverse order
        # until the start epoch is found
        for i, num_epoch in enumerate(num_epochs[::-1]):
            next_lv_beg_idx += num_epoch
            if ckpt_epoch < next_lv_beg_idx:
                # start_idx = i
                start_level = 5 - i
                break
            else:
                start_epoch -= num_epoch

        cum_total_epochs = ckpt_epoch
        if ckpt_epoch != 0 and start_level == 5 and start_epoch == 0:
            print('All epochs has been run already!!!')
            start_level = 0
            start_epoch = num_epochs[0]

        print(f'start level: {start_level}\tstart epoch: {start_epoch}')

        for level in range(start_level, -1, -1):
            end_epoch = num_epochs[level]
            self.is_last_epoch = False

            for epoch in range(start_epoch, end_epoch):
                print(f'Level{level}\tEpoch {epoch}/{end_epoch}')
                if epoch == end_epoch - 1:
                    self.is_last_epoch = True
                ###########################################
                # Train
                ###########################################
                mean_cost = self.train(epoch, level)
                self.logger_train.info(
                    f"Level{level} {epoch:4d},\t{mean_cost:.4f}")

                # cum_total_epochs is indicating the next epoch to run
                cum_total_epochs += 1

                # ckpt is created when epoch comes to the
                # cum_total_epochs is indicating the next epoch to run,
                # threfore, it needs to be decremented by 1
                if self.is_last_epoch or epoch == (end_epoch - 1) // 2:
                    self.ckpt_handler.save_ckpt(cum_total_epochs - 1)

            # level cleared
            level -= 1

        # close the child process that makes train batches
        self.dataset.close()

        #################################################################
        # Evaluation
        #################################################################
        print('Evaluation being processed ...')
        total_ssim, total_psnr = self.evaluate()
        self.logger_eval.info(
            f"SSIM: {total_ssim:.4f},\tPSNR: {total_psnr:.4f}")

    # get the output images for each level
    def run_model(self):
        print("---------------------------------------------------------")
        print("         Starting PyNET")
        print("---------------------------------------------------------")

        # close the child process that makes train batches
        self.dataset.close()

        # creating network
        self._build_graph()

        # ckpt_epoch becomes the cum start epoch
        self.ckpt_handler.restore_ckpt()
        print('Images being processed ...')
        total_ssim, total_psnr = self.evaluate(save_output=True)
        print(f"SSIM: {total_ssim:.4f},\tPSNR: {total_psnr:.4f}")


if __name__ == '__main__':
    i_n = ISPNet('dummy')
    # i_n.run()
    i_n.run_model()
    print("The End\n\n\n")
