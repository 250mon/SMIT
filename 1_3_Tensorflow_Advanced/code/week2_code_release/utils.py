from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2
import numpy as np


def show_image(images, labels):
    
    # for example, batch size 256
    # sq_row = 16
    # images = 256x28x28
    sq_row = int(np.sqrt(np.shape(images)[0]))
    images = np.reshape(images, (-1, 28, 28))
    
    total_image = []
    pos = 0
    
    for row in range(sq_row):
        row_img = []
        for col in range(sq_row):
            # final row_img = 16x28x28; 16 images
            row_img.append(images[pos])
            pos += 1
        # final total_image = 16x28x448; 16 row_imgs
        total_image.append(np.concatenate(row_img, axis=1))
        
    # show_img = 448x448
    show_img = np.concatenate(total_image, axis=0)
    print(f'(row_img={np.array(row_img).shape}, total_image={np.array(total_image).shape}, show_img={show_img.shape}\n')
    
    cv2.imshow('Batch Data', show_img)
    key = cv2.waitKey(0)
    
    return key


class mnist_data:
    
    def __init__(self, FLAGS):
        
        self.image_file = self._read_file(os.path.join(FLAGS.data_path, FLAGS.img_name))
        self.image_size = int((self.image_file.seek(0, 2) - 16.)/784.) #seek(offset, [where]), where(0:start, 1:current, 2:end)
        self.label_file = self._read_file(os.path.join(FLAGS.data_path, FLAGS.label_name))
        self.label_size = int((self.label_file.seek(0, 2) - 8.0))
        
        assert self.image_size == self.label_size
        
        self.batch_size = FLAGS.batch_size
        self.index = np.arange(self.image_size)
        self._shuffle()
        
    
    def _read_file(self, filename):
        return open(filename, 'rb')
    
    
    def next_batch(self):
        images = []
        labels = []
        
        for _ in range(self.batch_size):
            
            self.image_file.seek(16 + 784 * self.index[self.position], 0)
            self.label_file.seek(8 + self.index[self.position], 0)
            
            # images is filled with an array of ubyte type
            # labes is fille swith an ubyte
            images.append(np.fromfile(self.image_file, dtype=np.ubyte, count=784))
            labels.append(np.fromfile(self.label_file, dtype=np.ubyte, count=1))
            
            self.position += 1
            
            if self.position == self.image_size:
                self._shuffle()

        # covert from list to np array
        # -1 means that the size needs to be inferred
        batch_image = np.reshape(images, (-1,784))
        batch_label = np.reshape(labels, (-1,))
        
        return batch_image, batch_label
    
    def _shuffle(self):
        np.random.shuffle(self.index)
        self.position = 0
