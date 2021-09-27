from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf

from settings import Settings
from network_params import NetParams
import utils
import network_model
from util_ckpt import CheckPtHandler
from util_log import Loggers
import pdb

tf.disable_eager_execution()


class ICNet:
    def __init__(self, name):
        self.settings = Settings()
        self.settings.tb_log_dir = self.settings.tb_log_dir + '_' + name
        self.settings.ckpt_dir = self.settings.ckpt_dir + '_' + name
        # network params created
        self.net_params = NetParams(self)
        # dataset
        self.dataset = utils.CityscapesReader(self.settings)
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
        self.network = network_model.NetModel(self)
        # ckpt handler
        self.ckpt_handler = CheckPtHandler(self)

    def train(self, epoch):
        curr_lr = self.settings.get_lr(epoch)
        num_of_batches = self.dataset.num_of_batches
        mean_cost = 0.

        for step in range(num_of_batches):
            if step % 100 == 0:
                print(f"Step {step}/{num_of_batches} running...")

            batch_images, batch_labels = self.dataset.next_batch()
            feed_dict = {
                self.network.t_batch_img: batch_images,
                self.network.t_batch_lab: batch_labels,
                self.net_params.learning_rate_ph: curr_lr,
            }

            _, loss, outputs = self.network.session.run(
                (self.network.train_ops, self.network.loss, self.network.t_outputs),
                feed_dict=feed_dict)

            # pdb.set_trace()
            mean_cost += loss

            # if it is the last step of the level, then show images
            if step == num_of_batches - 1:
                self.validate_imgs(outputs, batch_labels, f'e{epoch:05d}')
        # calculate mean of losses of minibatch
        mean_cost /= float(num_of_batches)
        return mean_cost

    # show or/and save the label/output images of training
    def validate_imgs(self, outputs, labels, title):
        pass
        # v_min = outputs.min()
        # v_max = outputs.max()
        # outputs = ((outputs - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        #
        # resized_labels = self.dataset.resize_label(labels, outputs)
        #
        # util_datasets.show_zurich(outputs,
        #                           resized_labels,
        #                           title,
        #                           save=True)

    def evaluate(self, save_output=False):
        pass
        # eval_num = self.dataset.get_eval_number()
        # total_ssim = 0.
        # total_psnr = 0.
        # for step in range(eval_num):
        #     eval_images, eval_labels = self.dataset.next_eval_batch()
        #     feed_dict = {
        #         self.network.t_batch_img: eval_images,
        #         self.network.t_batch_lab: eval_labels,
        #     }
        #     # session run
        #     ssim, psnr, outputs = self.network.session.run(
        #         (self.network.t_loss_ops['eval_ssim'],
        #          self.network.t_loss_ops['eval_psnr'],
        #          self.network.t_outputs['lv1']),
        #         feed_dict=feed_dict)
        #     total_ssim += ssim
        #     total_psnr += psnr
        #     if save_output is True:
        #         self.validate_imgs(outputs, eval_labels, f'lv1_img{step:02d}')
        #
        # total_ssim /= eval_num
        # total_psnr /= eval_num
        # return total_ssim, total_psnr

    def run(self):
        print("---------------------------------------------------------")
        print("         Starting Zurich-Data Batch Processing Example")
        print("---------------------------------------------------------")

        # creating network
        self._build_graph()
        self.logger_train = self.loggers.create_logger('icnet_train')
        self.logger_eval = self.loggers.create_logger('icnet_eval')

        # ckpt_epoch becomes the cum start epoch
        ckpt_epoch = self.ckpt_handler.restore_ckpt()
        start_epoch = ckpt_epoch
        end_epoch = self.settings.num_epoch
        print(f'start epoch: {start_epoch}')
        for epoch in range(start_epoch, end_epoch):
            print(f'Epoch {epoch}/{end_epoch}')
            if epoch == end_epoch - 1:
                self.is_last_epoch = True
            ###########################################
            # Train
            ###########################################
            mean_cost = self.train(epoch, level)
            self.logger_train.info(f"{epoch:4d},\t{mean_cost:.4f}")

        # close the child process that makes train batches
        self.dataset.close()

        #################################################################
        # Evaluation
        #################################################################
        # print('Evaluation being processed ...')
        # accuracy = self.evaluate()
        # self.logger_eval.info(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    i_n = ICNet('dummy')
    print("The End\n\n\n")
