"""
Created on Sat Mar. 09 15:09:17 2019

@author: ygkim

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import utils
import model
import model_vgg16
import model_resnet
import logging
import util_log

FLAGS = None


def get_arguments():

    parser = argparse.ArgumentParser(
        'Implementation for CIFAR handwritten digits 2020')

    parser.add_argument('--num_epoch',
                        type=int,
                        default=50,
                        help='Parameter for learning rate',
                        required=False)

    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='parameter for batch size',
                        required=False)

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0001,
                        help='Parameter for learning rate',
                        required=False)

    parser.add_argument('--net_type',
                        type=str,
                        default='Resnet50',
                        help='Parameter for Network Selection',
                        required=False)

    return parser.parse_args()


def main():
    args = get_arguments()
    cfg = utils.DataPreprocesser(args)

    print("---------------------------------------------------------")
    print("         Starting CIFAR Batch Processing Example")
    print("---------------------------------------------------------")

    # Determine net type
    networks = {
        'Dense': model.DenseNet,
        'Conv': model.ConvNet,
        'VGG16': model_vgg16.VGG16Net,
        'Resnet34': model_resnet.ConvNet,
        'Resnet50': model_resnet.ConvNet,
    }
    net = networks[args.net_type](cfg)

    cfg.set_logging(cfg.net_type)
    # logger = logging.getLogger(cfg.net_type)

    # Read data
    # mnist_data = utils.MnistReader(cfg)
    # cifar = utils.Cifar100Reader(cfg)
    cifar = utils.Cifar10DataSets(cfg)

    # while 1:
    # batch_image, batch_label = cifar.next_batch()
    # print("batch generation,   PRESS any key to proceed and 'q' to quit this program")
    # #key = utils.show_mnist(batch_image, batch_label)
    # key = utils.show_cifar(batch_image, batch_label)
    # if key == ord('q'):
    # break

    _train_op, _loss, _logits = net.optimizer()

    #per_epoch = mnist.image_size//cfg.batch_size
    per_epoch = cifar.image_size // cfg.batch_size
    for epoch in range(cfg.num_epoch):

        mean_cost = 0.

        for step in range(per_epoch):
            print(f"Epoch {epoch}... Step {step} / {per_epoch} running...")
            # input tensor assignment
            # images, labels = mnist.next_batch()
            images, labels = cifar.next_batch()
            feed_dict = {net.t_batch_img: images, net.t_batch_lab: labels}

            _, loss = net.sess.run((_train_op, _loss), feed_dict=feed_dict)
            mean_cost += loss
            cfg.col_data.write_record(Step=step, Step_Cost=loss)

        mean_cost /= float(per_epoch)
        cfg.col_data.write_record(Epoch=epoch, Epoch_Cost=mean_cost)
        print("Learning at %d epoch " % epoch,
              "==== Cost of %1.8f" % mean_cost)
        # logger.info(f"{epoch}, {mean_cost}")

        #################################################################
        # EVALUATION
        #################################################################
        # feed_dict = {net.batch_img: mnist.eval_data[0], net.batch_lab: mnist.eval_data[1]}
        #ToDo: eval_data reshape
        # feed_dict = {net.batch_img: cifar.eval_data[0], net.batch_lab: cifar.eval_data[1]}
        # logits = net.sess.run(_logits, feed_dict=feed_dict)

        # correct_prediction = np.equal(np.argmax(logits, axis=1), mnist.eval_data[1])
        # correct_prediction = np.equal(np.argmax(logits, axis=1), cifar.eval_data[1])
        # accuracy = np.mean(correct_prediction)
        # print("Evaluation at %d epoch " %epoch, "==== Accuracy of %1.4f" %accuracy)


if __name__ == '__main__':

    main()
