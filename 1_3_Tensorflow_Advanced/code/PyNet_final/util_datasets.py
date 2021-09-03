from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2, glob, time, random
import numpy as np
import tensorflow.compat.v1 as tf
from multiprocessing import Process, Manager, Lock
import pdb

os.makedirs('imgs', exist_ok=True)


def show_zurich1(images, labels):
    sq_row = int(np.sqrt(np.shape(images)[0]))

    total_image = []
    total_image1 = []
    pos = 0

    for row in range(sq_row):
        row_img = []
        row_img1 = []
        for col in range(sq_row):
            row_img.append((images[pos, :, :, :3] // 4).astype(np.uint8))
            row_img1.append(labels[pos])
            pos += 1
        total_image.append(np.concatenate(row_img, axis=1))
        total_image1.append(np.concatenate(row_img1, axis=1))

    show_img = np.concatenate(total_image, axis=0)
    show_img1 = np.concatenate(total_image1, axis=0)

    cv2.imshow('Training Image', show_img)
    cv2.imshow('Label Image', show_img1)
    key = cv2.waitKey(0)

    return key


def show_zurich(images, labels, title, save=True):
    def _arrange_imgs(imgs):
        sq_row = int(np.sqrt(np.shape(imgs)[0]))

        if imgs.shape[-1] == 4:
            imgs = (imgs[:, :, :, :-1] // 4).astype(np.uint8)

        total_image = []
        pos = 0
        for row in range(sq_row):
            row_img = []
            for col in range(sq_row):
                row_img.append(imgs[pos])
                pos += 1
            total_image.append(np.concatenate(row_img, axis=1))
        show_img = np.concatenate(total_image, axis=0)
        return show_img

    pic_images = _arrange_imgs(images)
    title_images = title + '_huawei'
    pic_labels = _arrange_imgs(labels)
    title_labels = title + '_label'

    if save:
        filename = os.path.join('imgs', title_images + '.png')
        cv2.imwrite(filename, pic_images)
        filename = os.path.join('imgs', title_labels + '.png')
        cv2.imwrite(filename, pic_labels)
    else:
        cv2.imshow(title_images, pic_images)
        cv2.imshow(title_labels, pic_labels)

    key = cv2.waitKey(0)
    return key


class NeuralLoss:
    VGG19_DATA = './vgg-19-model.pb'
    VGG19_MAT = '../vgg-data/imagenet-vgg-verydeep-19.mat'
    NL_param = {
        'vgg_layers': [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
            'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
            'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
        ],
        'loss_layers': ['relu5_4'],
        'image_mean': [123.68, 116.779, 103.939]
    }

    def __init__(self):
        self.param = self.NL_param

        self.vgg_layers = self.param['vgg_layers']
        self.loss_layers = self.param['loss_layers']
        self.img_mean = np.array(self.param['image_mean'], dtype=np.float32)

        self.graph = self._get_default_graph()

    def _get_default_graph(self):

        with tf.gfile.GFile(self.VGG19_DATA, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            _ = tf.import_graph_def(graph_def, name='')

        return tf.get_default_graph()

    def get_layer_loss(self, img1, img2):

        img1 = tf.subtract(tf.cast(img1, tf.float32), self.img_mean)
        img2 = tf.subtract(tf.cast(img2, tf.float32), self.img_mean)

        nl_loss = 0.

        num_layers = float(len(self.loss_layers))

        for name in self.vgg_layers:

            kind = name[:4]

            if kind == 'conv':
                weight = self.graph.get_tensor_by_name(name + '/weight:0')
                bias = self.graph.get_tensor_by_name(name + '/bias:0')
                img1 = tf.nn.conv2d(
                    img1, weight, strides=(1, 1, 1, 1), padding='SAME') + bias
                img2 = tf.nn.conv2d(
                    img2, weight, strides=(1, 1, 1, 1), padding='SAME') + bias

                if name in self.loss_layers:
                    nl_loss += tf.reduce_mean(tf.square(img1 - img2))

            elif kind == 'relu':
                img1 = tf.nn.relu(img1)
                img2 = tf.nn.relu(img2)

                if name in self.loss_layers:
                    nl_loss += tf.reduce_mean(tf.square(img1 - img2))

            elif kind == 'pool':
                img1 = tf.nn.max_pool2d(img1,
                                        ksize=2,
                                        strides=2,
                                        padding='SAME')
                img2 = tf.nn.max_pool2d(img2,
                                        ksize=2,
                                        strides=2,
                                        padding='SAME')

                if name in self.loss_layers:
                    nl_loss += tf.reduce_mean(tf.square(img1 - img2))
            else:
                raise NotImplementedError(
                    '{:s} - No Such Layer in VGG-19 Net'.format(name))

        return nl_loss / num_layers


class PyNetReader(object):
    def __init__(self, isp_net):
        self.settings = isp_net.settings
        self.zurich_data = {
            'data_path': '../Z-Data',
            'train_name': 'train',
            'test_name': 'test'
        }

        self.buffer = Manager().list([])
        self.buffer_size = 10
        self.lock = Lock()
        self.end_flag = Manager().list([False])

        # Train images
        self.img_list = glob.glob(
            os.path.join(self.zurich_data['data_path'],
                         self.zurich_data['train_name'], 'huawei_raw', "*.*"))
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0
        self.num_epoch = self.settings.num_epoch
        self.batch_size = self.settings.batch_size
        self.num_of_batches = self.img_list_size // self.batch_size
        # Train labels
        self.label_path = os.path.join(self.zurich_data['data_path'],
                                       self.zurich_data['train_name'], 'canon')

        # Eval images
        self.eval_buffer = Manager().list([])
        self.eval_list = glob.glob(
            os.path.join(self.zurich_data['data_path'],
                         self.zurich_data['test_name'], 'huawei_raw', "*.*"))
        self.eval_list_size = len(self.eval_list)
        self.eval_list_pos = 0
        self.eval_batch_size = 1
        self.num_of_eval = self.eval_list_size // self.eval_batch_size
        # Eval labels
        self.eval_lab_path = os.path.join(self.zurich_data['data_path'],
                                          self.zurich_data['test_name'],
                                          'canon')

        self.p = Process(target=self._start_buffer)
        self.p.daemon = True
        self.p.start()
        time.sleep(0.5)

    # return an array of batch test images and an array of corresponding label images
    # by reading batch size of test images and label images successively
    def next_eval_batch(self):
        if self.eval_list_pos + self.eval_batch_size > self.eval_list_size:
            self.eval_list_pos = 0
        eval_cache = []
        lab_cache = []
        for index in range(self.eval_batch_size):
            img, lab = self._read_image(self.eval_list[self.eval_list_pos],
                                        self.eval_lab_path,
                                        augment=False)
            eval_cache.append(img)
            lab_cache.append(lab)
            self.eval_list_pos += 1
        eval_batch = np.concatenate(eval_cache, axis=0)
        lab_batch = np.concatenate(lab_cache, axis=0)
        return (eval_batch, lab_batch)

    # train buffer
    def _start_buffer(self):
        while (1):
            if self.end_flag[0]:
                break
            # get a list of a list of train batches(train, labels)
            _batch = self._get_batch()
            # keeps checking if there is any vacancy in the train buffer
            while (1):
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    time.sleep(0.5)
            self.lock.acquire()
            # if the buffer has some vacancy, it gets filled with a new list of batches
            self.buffer.append(_batch)
            self.lock.release()

    # get an array of batch train images and an array of corresponding label images
    # by reading batch size of train images and label images successively
    def _get_batch(self):
        if self.img_list_pos + self.batch_size > self.img_list_size - 1:
            self.img_list_pos = 0
            random.shuffle(self.img_list)
        tr_cache = []
        lab_cache = []
        for index in range(self.batch_size):
            tr, lab = self._read_image(self.img_list[self.img_list_pos],
                                       self.label_path,
                                       augment=True)
            tr_cache.append(tr)
            lab_cache.append(lab)
            self.img_list_pos += 1
        tr_batch = np.concatenate(tr_cache, axis=0)
        lab_batch = np.concatenate(lab_cache, axis=0)
        return (tr_batch, lab_batch)

    # read a train(or eval) and label image at a time and apply data aug
    # return a tupule of a train image and a label image
    def _read_image(self, path, lab_path, augment=True):
        # read train image
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 10-bit Bayer Pattern
        # convert 2 dim data to 3 dim data
        # raw (2H x 2W) -> tr_sample (H x W x C)
        # original image format:
        #   RGRG...
        #   GBGB...
        #   RGRG...
        #   .......
        ch_R = raw[0::2, 0::2]
        ch_Gb = raw[0::2, 1::2]
        ch_Gr = raw[1::2, 0::2]
        ch_B = raw[1::2, 1::2]
        tr_sample = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))

        # read label image; lab_path + 'tr_filename.png'[:-3] + 'jpg'
        lab_path = os.path.join(lab_path, os.path.basename(path))[:-3] + 'jpg'
        lab_sample = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)

        if augment:
            # image augmentation - random rotation and vertical flip
            rot_iter = np.random.randint(1, 100) % 4
            tr_sample = np.rot90(tr_sample, rot_iter)
            lab_sample = np.rot90(lab_sample, rot_iter)

            flip = np.random.randint(1, 100) % 2
            if flip:
                tr_sample = np.flipud(tr_sample)
                lab_sample = np.flipud(lab_sample)

        tr_sample = np.reshape(tr_sample, (1, ) + np.shape(tr_sample))
        lab_sample = np.reshape(lab_sample, (1, ) + np.shape(lab_sample))
        return (tr_sample, lab_sample)

    # match the size of a label image to the corresponding train image
    @staticmethod
    def resize_label(lab_imgs, tr_imgs):
        if tr_imgs.shape[-1] == 4:
            labels = np.zeros(tr_imgs[:, :, :, :-1].shape, dtype=np.uint8)
        else:
            labels = np.zeros(tr_imgs.shape, dtype=np.uint8)
        for idx, lab_img in enumerate(lab_imgs):
            labels[idx] = cv2.resize(lab_img,
                                     tr_imgs.shape[1:-1],
                                     interpolation=cv2.INTER_CUBIC)
        return labels

    # match the size of a label image to the corresponding train image
    @staticmethod
    def resize_label_by_dim(lab_imgs, height, weight):
        n, h, w, c = lab_imgs.shape
        labels = np.zeros((n, height, weight, c), dtype=np.uint8)
        for idx in range(len(lab_imgs)):
            labels[idx] = cv2.resize(lab_imgs[idx], (height, weight),
                                     interpolation=cv2.INTER_CUBIC)
        return labels

    # returns a list of batches(train, label) whose length is the batch size
    def next_batch(self):
        while (1):
            if len(self.buffer) == 0:
                time.sleep(0.1)
            else:
                self.lock.acquire()
                item = self.buffer.pop(0)
                self.lock.release()
                break
        return item

    def close(self):
        self.end_flag[0] = True

    # train number per epoch
    def get_num_of_batches(self):
        return self.num_of_batches

    # eval number
    def get_eval_number(self):
        return self.num_of_eval

    def change_batch_size(self, size):
        self.batch_size = size
        self.num_of_batches = self.img_list_size // self.batch_size
