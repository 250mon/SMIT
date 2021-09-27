from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2
import pickle
from abc import ABC, abstractmethod


class BatchRndIdxGen:
    ''' Generate random index range to make a batch of random order '''
    def __init__(self, max_idx, bshuffle=True):
        self.max_idx = max_idx
        self.bshuffle = bshuffle
        # a shuffled np array of an array [0, ... , max_idx-1]
        self.index = np.arange(self.max_idx)
        # a pointer which indicates the start of a chunk of index range for a batch
        self.beg = 0

    def get_next_indices(self, batch_size):
        if batch_size > self.max_idx:
            raise Exception('Batch size is greater than Data size')

        _end = self.beg + batch_size
        # if the end index gets out of the image total size,
        # set the beg to 0
        if _end > self.max_idx:
            self.beg = 0
            _end = self.beg + batch_size

        # whenever iteration begins, shuffle the indices
        if self.bshuffle and self.beg == 0:
            np.random.shuffle(self.index)

        next_idx_chunk = self.index[self.beg:_end]
        self.beg = _end
        return next_idx_chunk


"""
Data structure of original data:
    5 files consists of 10000 images foramted in 3 x 1032

Data paths to be processed:
    cifar10_data={
            'data_path': '../CIFAR/CIFAR-10',
            'train_files': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
            'eval_files': ['eval_batch']
            }

Data structure to be used:
    self.image_data; all train images (N x H x W x C)
    self.label_data; all train labels (N)
    image_size; N
    label_size; N

    self.eval_data; all eval images
    ...
"""


# Import dataset and reshape the org_data to the formatted data
# format:
#   image: N x H x W x C shape numpy array
#   label: N shape numpy array
class DataSets(ABC):
    def __init__(self, image_classifier):
        self.settings = image_classifier.settings
        self.net_params = image_classifier.net_params
        # dataset contain images and labels
        ''' format
        dataset = {
                'train': {
                    'images': N x H x W x C numpy array
                    'labels': N numpy array
                    }
                'eval': {
                    'images': N x H x W x C numpy array
                    'labels': N numpy array
                    }
                }
        '''
        self.dataset = {
            'train': {
                'images': None,
                'labels': None
            },
            'eval': {
                'images': None,
                'labels': None
            },
        }
        self._import_dataset()

        # for a train batch
        self.data_size = self.dataset['train']['images'].shape[0]
        self.num_epoch = self.settings.num_epoch
        self.batch_size = self.settings.batch_size
        self.num_of_batches = self.data_size // self.batch_size
        self.train_batch_idx_gen = BatchRndIdxGen(self.data_size)

        # for a eval batch
        self.eval_data_size = self.dataset['eval']['images'].shape[0]
        self.eval_batch_size = self.settings.eval_size
        self.eval_num_of_batches = self.eval_data_size // self.eval_batch_size
        self.eval_batch_idx_gen = BatchRndIdxGen(self.eval_data_size, bshuffle=False)

        # # for shuffling index to make a train batch
        # self.index = np.arange(self.get_size('train'))
        # self.position = 0
        # self._shuffle()

        # # for a eval batch
        # self.eval_size = self.settings.eval_size
        # self.eval_pos = 0

        # data augmentation
        self.au_min_scale = 0.75  #0.875 #(for 28x28 training)
        self.au_tar_size = np.multiply(self.au_min_scale, 32).astype(np.int32)
        self.au_max_scale = 1.25
        self.au_hor_flip = True

    # import dataset from a single file or multiple files
    # The whole data sets are stored into the self.dataset
    @abstractmethod
    def _import_dataset(self):
        pass

    # def get_data_size(self, type='train'):
        # return self.dataset[type]['images'].shape[0]

    # def get_images(self, type='train'):
        # return self.dataset[type]['images']

    # def get_labels(self, type='train'):
        # return self.dataset[type]['labels']

    # train number per epoch
    def get_num_of_batches(self):
        return self.num_of_batches

    # eval number
    def get_eval_number(self):
        return self.eval_num_of_batches

    # returns shuffled batch_images
    # mnist: (N x H x W x C) and labels
    # cifar: (N x H x W x C) and labels
    def next_train_batch(self):
        _idx_chunk = self.train_batch_idx_gen.get_next_indices(self.batch_size)
        batch_images = self.dataset['train']['images'][_idx_chunk]
        batch_labels = self.dataset['train']['labels'][_idx_chunk]
        batch_images = self._img_aug(batch_images)
        return batch_images, batch_labels

    def next_eval_batch(self):
        _idx_chunk = self.eval_batch_idx_gen.get_next_indices(self.eval_batch_size)
        batch_images = self.dataset['eval']['images'][_idx_chunk]
        batch_labels = self.dataset['eval']['labels'][_idx_chunk]
        return batch_images, batch_labels

    def _img_aug(self, batch):
        if self.net_params.use_data_aug is False:
            return batch
        """ Resizing and Flipping only """
        # scale +- 20%
        rand_scale = np.reshape(
            (self.au_max_scale - self.au_min_scale) *
            np.random.random(size=self.batch_size) + self.au_min_scale,
            (-1, 1))  # (batch_size, 1)
        orig_size = np.reshape(np.shape(batch)[1:3], (1, -1))  # (1, 2)

        new_size = np.matmul(rand_scale, orig_size).astype(np.int32)  # (batch_size, 2)
        changes = new_size - orig_size  #(batch_size, 2)
        token = changes[:, 0] >= 0
        changes = np.abs(changes)

        for idx in range(len(batch)):

            # flip horizontal
            if np.random.random() > 0.5:
                batch[idx] = cv2.flip(batch[idx], 1)
            '''
            # rotate 90 degree
            rot = np.random.random()
            if rot < 0.25:
                batch[idx] = np.rot90(batch[idx])
            elif rot < 0.5:
                batch[idx] = np.rot90(batch[idx], k=2)
            elif rot < 0.75:
                batch[idx] = np.rot90(batch[idx], k=3)
            else:
                rot = 0.
            '''
            temp = cv2.resize(batch[idx],
                              tuple(np.flip(new_size[idx])),
                              interpolation=cv2.INTER_LANCZOS4)

            if token[idx]:  # expanding size conversion
                crop_pos = np.random.random() * changes[idx]
                crop_pos = crop_pos.astype(np.int32)
                end_pos = crop_pos + orig_size[0]

                batch[idx] = temp[crop_pos[0]:end_pos[0],
                                  crop_pos[1]:end_pos[1], :]
            else:  # shrinking size conversion
                pad_s = int(np.random.random() *
                            changes[idx][0])  # shift effects
                pad_h = (pad_s, changes[idx][0] - pad_s)
                pad_s = int(np.random.random() * changes[idx][1])
                pad_w = (pad_s, changes[idx][1] - pad_s)

                batch[idx] = np.pad(temp, (pad_h, pad_w, (0, 0)), mode='edge')

        return batch

    # thumbnail preview of images
    # N x H x W x C format img data is needed
    def preview_images(self, type='train'):
        images = self.dataset[type]['images']
        sq_row = int(np.sqrt(np.shape(images)[0]))
        total_image = []
        pos = 0

        for row in range(sq_row):
            row_img = []
            for col in range(sq_row):
                row_img.append(images[pos])
                pos += 1
            total_image.append(np.concatenate(row_img, axis=1))

        show_img = cv2.cvtColor(np.concatenate(total_image, axis=0),
                                cv2.COLOR_RGB2BGR)

        cv2.imshow('Batch Data', show_img)
        key = cv2.waitKey(0)

        return key


''' dataset format
dataset = {
        'train': {
            'images': N x H x W x C numpy array
            'labels': N numpy array
            }
        'eval': {
            'images': N x H x W x C numpy array
            'labels': N numpy array
            }
        }
'''


class Cifar10DataSets(DataSets):
    def __init__(self, img_classifier):
        self.data_files = {
            'data_path':
            '../CIFAR/CIFAR-10',
            'train_files': [
                'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                'data_batch_5'
            ],
            'eval_files': ['test_batch']
        }
        super().__init__(img_classifier)

    def _import_dataset(self):
        self._import_dataset_sub('train')
        self._import_dataset_sub('eval')

    def _import_dataset_sub(self, type):
        _images = []
        _labels = []
        # 50000 images and labels imported into the lists
        for _filename in self.data_files[type + '_files']:
            _unpickled_dict = self._unpickle(
                os.path.join(self.data_files['data_path'], _filename))
            _images.append(_unpickled_dict['data'])
            _labels.append(_unpickled_dict['labels'])
        # np.reshape applied to 50000 images and labels
        # axis change; (N, C, H, W) to (N, H, W, C)
        self.dataset[type]['images'] = np.moveaxis(
            np.reshape(_images, (-1, 3, 32, 32)), 1, -1)
        self.dataset[type]['labels'] = np.reshape(_labels, (-1, ))

    def _unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict


class MnistDataSets(DataSets):
    def __init__(self, img_classifier):
        self.data_files = {
            'data_path': '../data',
            'image_file': 'train-images-idx3-ubyte',
            'label_file': 'train-labels-idx1-ubyte',
        }
        super().__init__(img_classifier)

    def _import_dataset(self):
        _images = []
        _labels = []

        with open(
                os.path.join(self.data_files['data_path'],
                             self.data_files['image_file']), 'rb') as fd:
            _data_read = np.fromfile(fd, dtype=np.ubyte)
            _images.append(_data_read[16:])

        _image_sets = np.reshape(_images, (-1, 28, 28, 1))
        _train_size = int(_image_sets.shape[0] * 0.8)
        self.dataset['train']['images'] = _image_sets[:_train_size, :, :, :]
        self.dataset['eval']['images'] = _image_sets[_train_size:, :, :, :]

        with open(
                os.path.join(self.data_files['data_path'],
                             self.data_files['label_file']), 'rb') as fd:
            _data_read = np.fromfile(fd, dtype=np.ubyte)
            _labels.append(_data_read[8:])

        _label_sets = np.reshape(_labels, (-1, ))
        self.dataset['train']['labels'] = _label_sets[:_train_size]
        self.dataset['eval']['labels'] = _label_sets[_train_size:]


if __name__ == '__main__':
    import image_classifier
    ic = image_classifier.ImageClassifier('normal')
    m = MnistDataSets(ic)
    m.preview_images(type='eval')
