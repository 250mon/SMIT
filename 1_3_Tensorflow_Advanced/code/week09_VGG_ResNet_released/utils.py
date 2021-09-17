from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2
import numpy as np

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
    
    weight_decay = 0.0005
        
    def __init__(self, args):
        print('Setup configurations...')
        
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.eval_size = args.eval_size
        self.learning_rate = args.learning_rate
        self.net_type = args.net_type
        self.ld_epoch = args.ld_epoch
        
        self.ckpt_dir = args.ckpt_dir
        self.log_dir = args.log_dir
        

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
        self.position = 0
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
        self.position = 0
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
        
        
"""
Data structure of original data:
    5 files consists of 10000 images foramted in 3 x 1032

Data paths to be processed:
    cifar10_data={
            'data_path': '../CIFAR/CIFAR-10',
            'train_name': ['data_batch_1', 'data_batch_2', 'data_batch_3',
                           'data_batch_4', 'data_batch_5'],
            'test_name': 'test_batch'}

Data structure to be used:
    self.image_data; all train images (N x H x W x C)
    self.label_data; all train labels (N)
    self.image_size; N
    self.label_size; N

    self.eval_data; all eval images
    ...
"""

class Cifar10Reader(object):
    
    def __init__(self, FLAGS):
        data = FLAGS.cifar10_data
        images = []
        labels = []
        
        # 50000 images and labels imported into the lists
        for idx in range(len(data['train_name'])):
            train = self._unpickle(os.path.join(data['data_path'], data['train_name'][idx]))
            images.append(train['data'])
            labels.append(train['labels'])

        # np.reshape applied to 50000 images and labels
        self.image_data = np.moveaxis(np.reshape(images, (-1, 3, 32, 32)), 1, -1)
        self.label_data = np.reshape(labels, (-1,))
        self.image_size = len(self.image_data)
        self.label_size = len(self.label_data)
        print('image data size - ', self.image_size)
        # eval data
        test = self._unpickle(os.path.join(data['data_path'], data['test_name']))
        self.eval_data = np.moveaxis(np.reshape(test['data'], (-1, 3, 32, 32)), 1, -1)
        self.eval_label = np.reshape(test['labels'], (-1))
        self.eval_size = FLAGS.eval_size
        self.eval_num = len(self.eval_data)//FLAGS.eval_size
        self.eval_pos = 0
        print('evaluation data size - ', len(self.eval_data))
        
        self.batch_size = FLAGS.batch_size
        # for shuffling index to make a mini batch
        self.index = np.arange(self.image_size)
        self.position = 0
        self._shuffle()

        # data augmentation
        self.au_min_scale = 0.75 #0.875 #(for 28x28 training)
        self.au_tar_size = np.multiply(self.au_min_scale, 32).astype(np.int32)
        self.au_max_scale = 1.25
        self.au_hor_flip = True
                
        
    def _unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    
    def next_batch(self):
        # container to hold next batch images and labels
        images = []
        labels = []
        
        for _ in range(self.batch_size):
            images.append(self.image_data[self.index[self.position]])
            labels.append(self.label_data[self.index[self.position]])
            self.position += 1
            
            if self.position == self.image_size:
                self._shuffle()
            
        batch_image = np.reshape(images, (-1, 32, 32, 3))
        batch_image = self._img_aug(batch_image)
        
        batch_label = np.reshape(labels, (-1, ))
                                                                
        return batch_image, batch_label
    
    
    def eval_batch(self):
        
        batch_image = self.eval_data[self.eval_pos*self.eval_size:(self.eval_pos+1)*self.eval_size, :, :, :]
        batch_label = self.eval_label[self.eval_pos*self.eval_size:(self.eval_pos+1)*self.eval_size]
        
        self.eval_pos = (self.eval_pos+1)%self.eval_num
        
        return batch_image, batch_label
    
        
    def _shuffle(self):
        np.random.shuffle(self.index)
        self.position = 0
        
    def _img_aug(self, batch):
        
        # scale +- 20%
        rand_scale = np.reshape((self.au_max_scale-self.au_min_scale)*np.random.random(size=self.batch_size)+self.au_min_scale, (-1, 1)) #(batch_size, 1)
        orig_size = np.reshape(np.shape(batch)[1:3], (1, -1)) # (1, 2)
        
        new_size = np.matmul(rand_scale, orig_size).astype(np.int32) #(batch_size, 2)
        changes = new_size - orig_size #(batch_size, 2)
        token = changes[:,0] >= 0
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
            temp = cv2.resize(batch[idx], tuple(np.flip(new_size[idx])), interpolation=cv2.INTER_LANCZOS4)
        
            if token[idx]: # expanding size conversion
                crop_pos = np.random.random() * changes[idx]
                crop_pos = crop_pos.astype(np.int32)
                end_pos = crop_pos + orig_size[0]
                
                batch[idx] = temp[crop_pos[0]:end_pos[0], crop_pos[1]:end_pos[1], :]
            else: # shrinking size conversion
                pad_s = int(np.random.random()*changes[idx][0]) # shift effects
                pad_h = (pad_s, changes[idx][0]-pad_s)
                pad_s = int(np.random.random()*changes[idx][1])
                pad_w = (pad_s, changes[idx][1]-pad_s)
                
                batch[idx] = np.pad(temp, (pad_h, pad_w, (0,0)), mode='edge')
                                        
        return batch
    
            

class LRController(object):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
                
        self.decay_epoch = cfg.ld_epoch
        self.decay_width = cfg.num_epoch - self.decay_epoch
        self.learning_rate = cfg.learning_rate
            
    def get_lr(self, epoch):
                
        if epoch < self.decay_epoch:
            lr =  self.learning_rate
        else:
            lr = 0.5 * self.learning_rate * (1. - float(epoch - self.decay_epoch)/float(self.decay_width))
            print('Linear Decay - Learning Rate is {:1.8f}'.format(lr))
            
        return lr
