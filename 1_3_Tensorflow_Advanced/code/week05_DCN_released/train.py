# -*- coding: utf-8 -*-
"""
Created on Mon Feb  17 13:17:50 2020

@author: Angelo
"""

import argparse
import utils
import model
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def get_arguments():
    parser = argparse.ArgumentParser(
        'Implementation for MNIST handwritten digits 2020')
    parser.add_argument('--num_epoch',
                        type=int,
                        default=10,
                        help='Parameter for learning rate',
                        required=False)
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='parameter for batch size',
                        required=False)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Parameter for learning rate',
                        required=False)
    parser.add_argument('--net_type',
                        type=str,
                        default='Dense',
                        help='Parameter for Network Selection',
                        required=False)
    return parser.parse_args()


def main():
    args = get_arguments()
    cfg = utils.DataPreprocesser(args)

    # for logging of loss and accuracy to tune the h-params
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = "logs/CNN/" + current_time + "/train"
    # test_log_dir = "logs/CNN/" + current_time + "/test"
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    # train_writer_flush = train_summary_writer.flush()
    # test_writer_flush = test_summary_writer.flush()

    print("---------------------------------------------------------")
    print("          Starting MNIST Mini-Batch Training")
    print("---------------------------------------------------------")
    mnist = utils.MnistReader(cfg)

    if cfg.net_type == 'Dense':
        net = model.DenseNet(cfg)
    elif cfg.net_type == 'Conv':
        net = model.ConvNet(cfg)
    else:
        raise NotImplementedError('Network Type is Not Defined')

    _train_op, _loss, _logits = net.optimizer()
    # net.sess.run([train_summary_writer.init(), test_summary_writer.init()])
    per_epoch = mnist.image_size // cfg.batch_size
    for epoch in range(cfg.num_epoch):
        mean_cost = 0.
        accuracy = 0.
        for step in range(per_epoch):

            images, labels = mnist.next_batch()
            feed_dict = {net.batch_img: images, net.batch_lab: labels}
            _, loss, logits = net.sess.run((_train_op, _loss, _logits),
                                           feed_dict=feed_dict)
            # loss per step
            mean_cost += loss
            # accuracy per step
            correct_prediction = np.equal(np.argmax(logits, axis=1), labels)
            accuracy += np.mean(correct_prediction)

        # loss per epoch
        mean_cost /= float(per_epoch)
        # accuracy per epoch
        accuracy /= float(per_epoch)

        # with train_summary_writer.as_default():
        # tf.summary.scalar('loss', mean_cost, step=epoch)
        # tf.summary.scalar('accuracy', accuracy, step=epoch)

        print("Training at %d epoch " % epoch,
              "\t==== Cost of %1.4f" % mean_cost)
        print("Training at %d epoch " % epoch,
              "\t==== Accuracy of %1.4f" % accuracy)

        #################################################################
        # EVALUATION
        #################################################################
        feed_dict = {
            net.batch_img: mnist.eval_data[0],
            net.batch_lab: mnist.eval_data[1]
        }
        loss, logits = net.sess.run((_loss, _logits), feed_dict=feed_dict)

        correct_prediction = np.equal(np.argmax(logits, axis=1),
                                      mnist.eval_data[1])
        accuracy = np.mean(correct_prediction)

        # with test_summary_writer.as_default():
        # tf.summary.scalar('loss', loss, step=epoch)
        # tf.summary.scalar('accuracy', accuracy, step=epoch)

        # net.sess.run([train_writer_flush, test_writer_flush])

        print("Evaluation at %d epoch " % epoch,
              "\t==== Accuracy of %1.4f" % accuracy)
        print("\n")


if __name__ == '__main__':
    main()
