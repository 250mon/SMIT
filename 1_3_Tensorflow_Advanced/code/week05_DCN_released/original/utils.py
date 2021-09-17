from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2
import numpy as np

class Config(object):
    
    data_path = '../data'
    img_name = 'train-images.idx3-ubyte'
    label_name = 'train-labels.idx1-ubyte'
    eval_part = 0.2
    
    def __init__(self, args):
        print('Setup configurations...')
        
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.net_type = args.net_type
        

def show_image(images, labels):
    
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


class MnistReader:
    
    def __init__(self, FLAGS):
        
        self.image_file = self._read_file(os.path.join(FLAGS.data_path, FLAGS.img_name))
        self.image_size = int((1.-FLAGS.eval_part)*(self.image_file.seek(0, 2) - 16.)/784.) #seek(offset, [where]), where(0:start, 1:current, 2:end)
        self.label_file = self._read_file(os.path.join(FLAGS.data_path, FLAGS.label_name))
        self.label_size = int((self.label_file.seek(0, 2) - 8.0))
        
        self.eval_data = self._get_evaldata()
        
        self.batch_size = FLAGS.batch_size
        self.index = np.arange(self.image_size)
        self._shuffle()
        
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
    
        
    def _read_file(self, filename):
        return open(filename, 'rb')
    
    
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