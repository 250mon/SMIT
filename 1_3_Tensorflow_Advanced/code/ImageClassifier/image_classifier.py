"""
Created on Sat Mar. 09 15:09:17 2019

@author: ygkim

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf
import argparse
import logging
import time
from datetime import timedelta

from settings import Settings
from network_params import NetParams
from util_datasets import Cifar10DataSets, MnistDataSets
from util_ckpt import CheckPtHandler
import model_dense
import model_conv
import model_vgg16
import model_resnet
from util_log import Loggers
import pdb

tf.disable_eager_execution()


class ImageClassifier:
    def __init__(self, name):
        self.settings = Settings()
        self.settings.tb_log_dir = self.settings.tb_log_dir + '_' + name
        self.settings.ckpt_dir = self.settings.ckpt_dir + '_' + name
        # network params created
        self.net_params = NetParams(self)
        # import dataset
        self.dataset = self._get_dataset()
        # tf graph related
        self.network = None
        self.ckpt_handler = None
        # for collecting analysis data (csv file)
        # self.analysis = util_log.AnalysisLog(logfile_name)
        # log file and analysis file name will be the network type
        self.loggers = Loggers(name)
        # for local loggings
        self.logger_train = None
        self.logger_eval = None

    def _build_graph(self):
        net_type = self.settings.net_type
        if net_type == 'VGG16':
            self.net_params.weight_decay = 0.0005
        elif net_type == 'Resnet34':
            self.net_params.weight_decay = 0.0002

        # build a graph by creating network
        # if any net-param needs to be changed,
        # it should be done before the build
        self.network = self._get_network()
        # chpt handler
        self.ckpt_handler = CheckPtHandler(self)

    def _get_network(self):
        net_type = self.settings.net_type
        networks = {
            'Dense': model_dense.DenseNet,
            'Conv': model_conv.ConvNet,
            'VGG16': model_vgg16.VGG16Net,
            'Resnet34': model_resnet.ResNet34,
            'Resnet50': model_resnet.ResNet50,
        }
        net = networks.get(net_type, None)
        if net is not None:
            net = net(self)
        else:
            raise NotImplementedError('Network Type is Not Defined')

        return net

    def change_network(self, net_type):
        tf.reset_default_graph()
        self.settings.net_type = net_type
        self.network = self._get_network()

    def _get_dataset(self):
        selected_set = self.settings.dataset
        datasets = {
            'cifar10': Cifar10DataSets,
            'mnist': MnistDataSets,
        }
        dataset = datasets.get(selected_set, None)
        if dataset is not None:
            dataset = dataset(self)
        else:
            raise NotImplementedError('Data set is Not Defined')
        return dataset

    def change_dataset(self, dataset):
        self.settings.dataset = dataset
        self.dataset = self._get_dataset()

    def _partial_feed_dict(self, epoch, proc='train'):
        # for batch norm
        if self.net_params.use_batch_norm:
            if proc == 'eval':
                train_bn = False
                reset_bn = False
            elif epoch % 2 == 0:
                train_bn = True
                reset_bn = True
            else:
                train_bn = True
                reset_bn = False
            batch_dict = {
                self.net_params.ph_bn_reset: reset_bn,
                self.net_params.ph_bn_train: train_bn,
            }
        else:
            batch_dict = {}

        # for drop out
        if self.net_params.use_drop:
            if proc == 'eval':
                use_drop = False
            elif epoch > 100:
                use_drop = False
            else:
                use_drop = True
            drop_dict = {
                self.net_params.ph_use_drop: use_drop,
            }
        else:
            drop_dict = {}

        return {**batch_dict, **drop_dict}

    def train(self, epoch):
        curr_lr = self.settings.get_lr(epoch)
        p_feed_dict = self._partial_feed_dict(epoch)

        num_of_batches = self.dataset.get_num_of_batches()
        mean_cost_wd = 0.
        mean_cost = 0.

        for step in range(num_of_batches):
            # if step % 10 == 0:
            # print(f"Step {step}/{num_of_batches} running...")
            images, labels = self.dataset.next_train_batch()
            feed_dict = {
                self.network.t_batch_img: images,
                self.network.t_batch_lab: labels,
                self.net_params.ph_learning_rate: curr_lr,
                **p_feed_dict,
            }
            # session run
            _, loss_wd, loss = self.network.session.run(
                (self.network.train_op, self.network.loss_wd,
                 self.network.loss),
                feed_dict=feed_dict)

            mean_cost_wd += loss_wd
            mean_cost += loss
        # calculate mean of losses of minibatch
        mean_cost_wd /= float(num_of_batches)
        mean_cost /= float(num_of_batches)
        return mean_cost_wd, mean_cost

    def evaluate(self, epoch):
        p_feed_dict = self._partial_feed_dict(epoch, proc='eval')

        eval_num = self.dataset.get_eval_number()
        total_accuracy = 0.
        for step in range(eval_num):
            eval_images, eval_labels = self.dataset.next_eval_batch()
            feed_dict = {
                self.network.t_batch_img: eval_images,
                self.network.t_batch_lab: eval_labels,
                **p_feed_dict,
            }
            # session run
            predicted_labels = self.network.session.run(
                self.network.t_predicted_labels, feed_dict=feed_dict)

            correct_prediction = np.equal(predicted_labels, eval_labels)
            total_accuracy += np.mean(correct_prediction)
        total_accuracy /= eval_num
        return total_accuracy

    def run(self):
        tf.reset_default_graph()
        # creating network
        self._build_graph()
        self.logger_train = self.loggers.create_logger(self.settings.net_type +
                                                       '_train')
        self.logger_eval = self.loggers.create_logger(self.settings.net_type +
                                                      '_eval')

        start_time = time.time()
        total_accuracy = 0.
        max_accuracy = 0.

        start_epoch = self.ckpt_handler.restore_ckpt()
        end_epoch = self.settings.num_epoch
        for epoch in range(start_epoch, end_epoch):
            # print(f'Epoch {epoch}/{end_epoch}')
            #################################################################
            # Train
            #################################################################
            mean_cost_wd, mean_cost = self.train(epoch)

            self.logger_train.info(
                f"{epoch:4d},\t{mean_cost_wd:.4f},\t{mean_cost:.4f}")
            #################################################################
            # Evaluation
            #################################################################
            if (epoch + 1) % 4 == 0:
                total_accuracy = self.evaluate(epoch)
                lap = time.time() - start_time
                lap_str = str(timedelta(seconds=lap))
                self.logger_eval.info(
                    f"{epoch:4d},\t{total_accuracy:.4f},\t{max_accuracy:.4f},\t{lap_str}"
                )

            ###############################################################
            # Summary and Ceckpoint
            # ################################################################
            feed_dict = {
                self.network.sum_losses:
                (mean_cost_wd, mean_cost, total_accuracy)
            }
            summaries = self.network.session.run(self.network.summaries,
                                                 feed_dict=feed_dict)
            self.network.summary_writer.add_summary(summaries, epoch)

            if total_accuracy > max_accuracy:
                self.ckpt_handler.save_ckpt(epoch)
                max_accuracy = total_accuracy


def print_mode(mode):
    msg_row_len = 40
    lside = 10
    mode_len = len(mode)
    msg_top_border = '#' * msg_row_len
    msg_lside_border = '#' * lside
    msg_rside_border = '#' * (msg_row_len - lside - mode_len - 2)
    print(msg_top_border)
    print(msg_lside_border + ' ' + mode + ' ' + msg_rside_border)
    print(msg_top_border)


def ic_set_for_mnist(mode):
    mode = 'm_' + mode
    print_mode(mode)
    ic = ImageClassifier(mode)
    ic.change_dataset('mnist')
    # ic.change_network('Dense')
    ic.change_network('Conv')
    ic.settings.num_epoch = 100
    ic.settings.batch_size = 256
    ic.settings.eval_size = 512
    ic.settings.learning_rate = 0.001
    ic.net_params.use_batch_norm = False
    ic.net_params.use_weight_decay = False
    ic.net_params.use_drop = False
    ic.net_params.use_data_aug = False
    return ic


def mnist_grid_search():

    print("---------------------------------------------------------")
    print("         Starting Mnist Batch Processing Example")
    print("---------------------------------------------------------")

    # batch 256, ReLu, he_normal
    mode = 'normal'
    ic = ic_set_for_mnist(mode)
    ic.net_params.batch_size = 256
    ic.run()

    # Activation
    mode = 'LReLu'
    ic = ic_set_for_mnist(mode)
    ic.net_params.activation = mode
    ic.run()

    mode = 'PReLu'
    ic = ic_set_for_mnist(mode)
    ic.net_params.activation = mode
    ic.run()

    # mode = 'SWISH'
    # ic = ic_set_for_mnist(mode)
    # ic.net_params.activation = mode
    # ic.run()

    # batch size
    mode = 'batch_64'
    ic = ic_set_for_mnist(mode)
    ic.net_params.batch_size = 64
    ic.run()

    # batch size
    mode = 'batch_1024'
    ic = ic_set_for_mnist(mode)
    ic.net_params.batch_size = 1024
    ic.run()

    # initializer
    mode = 'glorot_normal'
    ic = ic_set_for_mnist(mode)
    ic.net_params.initializer = mode
    ic.run()

    mode = 'glorot_uniform'
    ic = ic_set_for_mnist(mode)
    ic.net_params.initializer = mode
    ic.run()

    mode = 'he_uniform'
    ic = ic_set_for_mnist(mode)
    ic.net_params.initializer = mode
    ic.run()

    mode = 'glorot_normal'
    ic = ic_set_for_mnist(mode)
    ic.net_params.initializer = mode
    ic.run()

    # learning rate
    mode = 'lr_0.01'
    ic = ic_set_for_mnist(mode)
    ic.settings.learning_rate = 0.01
    ic.run()

    mode = 'lr_0.1'
    ic = ic_set_for_mnist(mode)
    ic.settings.learning_rate = 0.1
    ic.run()

    mode = 'lr_b1_0_b2_0.9'
    ic = ic_set_for_mnist(mode)
    ic.net_params.adabeta1 = 0
    ic.net_params.adabeta2 = 0.9
    ic.run()

    mode = 'lr_b1_0.1_b2_0.9'
    ic = ic_set_for_mnist(mode)
    ic.net_params.adam_beta1 = 0.1
    ic.net_params.adam_beta2 = 0.9
    ic.run()

    mode = 'lr_b1_0.5_b2_0.9'
    ic = ic_set_for_mnist(mode)
    ic.net_params.adam_beta1 = 0.5
    ic.net_params.adam_beta2 = 0.9
    ic.run()


def cifar_grid_search():
    print("---------------------------------------------------------")
    print("         Starting CIFAR Batch Processing Example")
    print("---------------------------------------------------------")

    mode = 'c_normal'
    print_mode(mode)
    ic = ImageClassifier(mode)
    ic.run()

    mode = 'c_wd_off'
    print_mode(mode)
    ic3 = ImageClassifier(mode)
    ic3.net_params.use_weight_decay = False
    ic3.run()

    mode = 'c_bn_off'
    print_mode(mode)
    ic1 = ImageClassifier(mode)
    ic1.net_params.use_batch_norm = False
    ic1.run()

    mode = 'c_aug_off'
    print_mode(mode)
    ic2 = ImageClassifier(mode)
    ic2.net_params.use_data_aug = False
    ic2.run()

    mode = 'c_do_off'
    print_mode(mode)
    ic4 = ImageClassifier(mode)
    ic4.net_params.use_drop = False
    ic4.run()


if __name__ == '__main__':
    import sys
    ch = sys.argv.pop(1)
    if ch == 'm':
        mnist_grid_search()
    else:
        cifar_grid_search()
