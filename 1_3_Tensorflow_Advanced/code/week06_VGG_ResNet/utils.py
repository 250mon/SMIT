from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2
import numpy as np
import util_log
import logging


class Config(object):
    
    mnist_data={
            'data_path': '../data',
            'img_name': 'train-images.idx3-ubyte',
            'label_name': 'train-labels.idx1-ubyte',
            'eval_part': 0.2}
    
    cifar100_data={
            'data_path': '../CIFAR/CIFAR-100',
            'train_name': 'train',
            'test_name': 'test'}
    
    cifar10_data={
            'data_path': '../CIFAR/CIFAR-10',
            'train_name': ['data_batch_1', 'data_batch_2', 'data_batch_3',
                           'data_batch_4', 'data_batch_5'],
            'test_name': 'test_batch'}
    
    def __init__(self, args):
        print('Setup configurations...')
        
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.net_type = args.net_type
        # for collecting data
        self.col_data = util_log.AnalysisLog(self.net_type)

    def set_logging(self, log_file, f_log_lvl=logging.INFO, c_log_lvl=logging.INFO):
        util_log.set_logging(log_file, f_log_lvl, c_log_lvl)
        

def show_mnist(images, labels):
    
    sq_row = int(np.sqrt(np.shape(images)[0]))
    images = np.reshape(images, (-1, 28, 28))
    
    total_image = []
    pos = 0
    
    for row in range(sq_row):
        row_img = []
        for col in range(sq_row):
            row_img.append(images[pos])
            pos += 1
        total_image.append(np.concatenate(row_img, axis=1))
        
    show_img = np.concatenate(total_image, axis=0)
    
    cv2.imshow('Batch Data', show_img)
    key = cv2.waitKey(0)
    
    return key

def show_cifar(images, labels):
    
    sq_row = int(np.sqrt(np.shape(images)[0]))
        
    total_image = []
    pos = 0
    
    for row in range(sq_row):
        row_img = []
        for col in range(sq_row):
            row_img.append(images[pos])
            pos += 1
        total_image.append(np.concatenate(row_img, axis=1))
    
    show_img = cv2.cvtColor(np.concatenate(total_image, axis=0), cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Batch Data', show_img)
    key = cv2.waitKey(0)
    
    return key


class MnistReader(object):
    
    def __init__(self, FLAGS):
        
        self.image_file = self._read_file(os.path.join(FLAGS.mnist_data['data_path'], FLAGS.mnist_data['img_name']))
        self.image_size = int((1.-FLAGS.mnist_data['eval_part'])*(self.image_file.seek(0, 2) - 16.)/784.) #seek(offset, [where]), where(0:start, 1:current, 2:end)
        self.label_file = self._read_file(os.path.join(FLAGS.mnist_data['data_path'], FLAGS.mnist_data['label_name']))
        self.label_size = int((self.label_file.seek(0, 2) - 8.0))
        
        self.eval_data = self._get_evaldata()
        
        self.batch_size = FLAGS.batch_size
        self.index = np.arange(self.image_size)
        self._shuffle()
        
        
    def _read_file(self, filename):
        return open(filename, 'rb')
        
    def _get_evaldata(self):
    
        images = []
        labels = []
        
        for idx in range(self.image_size, self.label_size):
            
            self.image_file.seek(16 + 784 * idx, 0)
            self.label_file.seek(8 + idx, 0)
            
            images.append(np.fromfile(self.image_file, dtype=np.ubyte, count=784))
            labels.append(np.fromfile(self.label_file, dtype=np.ubyte, count=1))
                                    
        batch_image = np.reshape(images, (-1, 784))
        batch_label = np.reshape(labels, (-1, ))
                                                        
        return (batch_image, batch_label)
    
    def next_batch(self):
        images = []
        labels = []
        
        for _ in range(self.batch_size):
            
            self.image_file.seek(16 + 784 * self.index[self.position], 0)
            self.label_file.seek(8 + self.index[self.position], 0)
            
            images.append(np.fromfile(self.image_file, dtype=np.ubyte, count=784))
            labels.append(np.fromfile(self.label_file, dtype=np.ubyte, count=1))
            
            self.position += 1
            
            if self.position == self.image_size:
                self._shuffle()
            
        batch_image = np.reshape(images, (-1, 784))
        batch_label = np.reshape(labels, (-1, ))
                                                                
        return batch_image, batch_label
    
    def _shuffle(self):
        np.random.shuffle(self.index)
        self.position = 0
    


class Cifar100Reader(object):
    
    def __init__(self, FLAGS):
        
        data = FLAGS.cifar100_data
        train = self._unpickle(os.path.join(data['data_path'], data['train_name']))
        test = self._unpickle(os.path.join(data['data_path'], data['test_name']))
        
        self.image_data = train['data']
        self.image_size = len(self.image_data)
        self.label_data = train['fine_labels']
        self.label_size = len(self.label_data)
        
        self.eval_data = test['data']
        self.eval_label = test['fine_labels']
        
        self.batch_size = FLAGS.batch_size
        self.index = np.arange(self.image_size)
        self._shuffle()
                
        
    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    
    def next_batch(self):
        images = []
        labels = []
        
        for _ in range(self.batch_size):
            
            image = np.transpose(np.reshape(self.image_data[self.index[self.position]], (3, -1)))
            images.append(np.reshape(image, (32, 32, 3)))
            labels.append(self.label_data[self.index[self.position]])
            
            self.position += 1
            
            if self.position == self.image_size:
                self._shuffle()
            
        batch_image = np.reshape(images, (-1, 32, 32, 3))
        batch_label = np.reshape(labels, (-1, ))
                                                                
        return batch_image, batch_label
    
    
    def _shuffle(self):
        np.random.shuffle(self.index)
        self.position = 0
        
        
class Cifar10Reader(object):
    
    def __init__(self, FLAGS):
        
        data = FLAGS.cifar10_data
        images = []
        labels = []
        
        for idx in range(len(data['train_name'])):
            train = self._unpickle(os.path.join(data['data_path'], data['train_name'][idx]))
            images.append(train['data'])
            labels.append(train['labels'])
            
        self.image_data = np.reshape(images, (-1, 3072))
        self.label_data = np.reshape(labels, (-1,))
        self.image_size = len(self.image_data)
        self.label_size = len(self.label_data)
        print('image data size - ', self.image_size)
        test = self._unpickle(os.path.join(data['data_path'], data['test_name']))
        self.eval_data = test['data']
        self.eval_label = test['labels']
        
        self.batch_size = FLAGS.batch_size
        self.index = np.arange(self.image_size)
        self._shuffle()
                
        
    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    
    def next_batch(self):
        images = []
        labels = []
        
        for _ in range(self.batch_size):
            
            image = np.transpose(np.reshape(self.image_data[self.index[self.position]], (3, -1)))
            images.append(np.reshape(image, (32, 32, 3)))
            labels.append(self.label_data[self.index[self.position]])
            
            self.position += 1
            
            if self.position == self.image_size:
                self._shuffle()
            
        batch_image = np.reshape(images, (-1, 32, 32, 3))
        batch_label = np.reshape(labels, (-1, ))
                                                                
        return batch_image, batch_label
    
    
    def _shuffle(self):
        np.random.shuffle(self.index)
        self.position = 0
