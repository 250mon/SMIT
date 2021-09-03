from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, cv2, glob, time, random, scipy.io
import numpy as np
import tensorflow.compat.v1 as tf
from multiprocessing import Process, Manager, Lock

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
    
    zurich_data={
            'data_path': '../../Z-Data',
            'train_name': 'train',
            'test_name': 'test'}
    
    VGG19_DATA = './vgg-19-model.pb'
    VGG19_MAT = '../vgg-data/imagenet-vgg-verydeep-19.mat'
    NL_param = {'vgg_layers': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 
                        'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 
                        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
                        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                        'relu5_3', 'conv5_4', 'relu5_4'],
                'loss_layers': ['relu5_4'],
                'image_mean': [123.68,  116.779,  103.939]}
    
    weight_decay = 0.0005
    
    num_epoch = 400
    batch_size = 16
    learning_rate = 0.0005
    
    BUFFER_SIZE = 10
            
    def __init__(self, args):
        print('Setup configurations...')
        
        self.net_type = args.net_type
        self.ckpt_dir = args.ckpt_dir
        self.log_dir = args.log_dir
        self.res_dir = args.res_dir
        

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

def show_zurich(images, labels):
    
    sq_row = int(np.sqrt(np.shape(images)[0]))
        
    total_image = []
    total_image1 = []
    pos = 0
    
    for row in range(sq_row):
        row_img = []
        row_img1 = []
        for col in range(sq_row):
            row_img.append((images[pos,:,:,:3]//4).astype(np.uint8))
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



class NeuralLoss(object):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.param = cfg.NL_param
        
        self.vgg_layers = self.param['vgg_layers']
        self.loss_layers = self.param['loss_layers']
        self.img_mean = np.array(self.param['image_mean'], dtype=np.float32)
        
        self.graph = self._get_default_graph()
    
    def _get_default_graph(self):
        
        with tf.gfile.GFile(self.cfg.VGG19_DATA, 'rb') as f:
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
                img1 = tf.nn.conv2d(img1, weight, strides=(1, 1, 1, 1), padding='SAME') + bias
                img2 = tf.nn.conv2d(img2, weight, strides=(1, 1, 1, 1), padding='SAME') + bias
                
                if name in self.loss_layers:
                    nl_loss += tf.reduce_mean(tf.square(img1-img2))
            
            elif kind == 'relu':
                img1 = tf.nn.relu(img1)
                img2 = tf.nn.relu(img2)
                
                if name in self.loss_layers:
                    nl_loss += tf.reduce_mean(tf.square(img1-img2))
            
            elif kind == 'pool':
                img1 = tf.nn.max_pool2d(img1, ksize=2, strides=2, padding='SAME')
                img2 = tf.nn.max_pool2d(img2, ksize=2, strides=2, padding='SAME')
                
                if name in self.loss_layers:
                    nl_loss += tf.reduce_mean(tf.square(img1-img2))
                #img = tf.nn.max_pool2d(img, ksize=2, strides=2, padding='VALID', name=name)                
            else:
                raise NotImplementedError('{:s} - No Such Layer in VGG-19 Net'.format(name))
        
                    
        return nl_loss/num_layers



class PyNetReader(object):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        
        self.buffer = Manager().list([])
        self.buffer_size = cfg.BUFFER_SIZE
        self.lock = Lock()
        self.end_flag = Manager().list([False])
        
        self.batch_size = cfg.batch_size
        
        self.img_list = glob.glob(os.path.join(cfg.zurich_data['data_path'], cfg.zurich_data['train_name'], 'huawei_raw', "*.*"))
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0
        
        self.label_path = os.path.join(cfg.zurich_data['data_path'], cfg.zurich_data['train_name'], 'canon')
        
        self.eval_list = glob.glob(os.path.join(cfg.zurich_data['data_path'], cfg.zurich_data['test_name'], 'huawei_raw', "*.*"))
        self.eval_lab = os.path.join(cfg.zurich_data['data_path'], cfg.zurich_data['test_name'], 'canon')
        self.eval_buffer = Manager().list([])
        
                
        self.res_dir = cfg.res_dir
        
        self.e = Process(target=self._start_eval)
        self.e.daemon=True
        self.e.start()
        time.sleep(0.5)
        
        
        self.p = Process(target=self._start_buffer)
        self.p.daemon=True
        self.p.start()
        time.sleep(0.5)
    
    
    def _start_eval(self):
        
        st_time = time.time()
        for idx in range(len(self.eval_list)):
            self.eval_buffer.append(self._read_image(self.eval_list[idx], self.eval_lab, augment=False))
            
        print('Evaluation Data Processing ...... Done - elapsed {:.4f}sec'.format(time.time()-st_time))
    
    
    def _start_buffer(self):
        
        while(1):
            
            if self.end_flag[0]:
                break
            
            _batch = self._get_batch()
                        
            while(1):
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    time.sleep(0.5)
                    
            self.lock.acquire()
            self.buffer.append(_batch)
            self.lock.release()
            #print('Stuffed - Buffer Size  {:d}'.format(len(self.buffer)))
            
    def _get_batch(self):
        
        if self.img_list_pos + self.batch_size > self.img_list_size-1:
            self.img_list_pos = 0
            random.shuffle(self.img_list)
        
        tr_cache = []
        lab_cache = []
                
        for index in range(self.batch_size):
            tr, lab = self._read_image(self.img_list[self.img_list_pos], self.label_path, augment=True)
            
            tr_cache.append(tr)
            lab_cache.append(lab)
            
            self.img_list_pos += 1
                
        tr_batch = np.concatenate(tr_cache, axis=0)
        lab_batch = np.concatenate(lab_cache, axis=0)
       
        return (tr_batch, lab_batch)
    
    def _read_image(self, path, lab_path, augment=True):
        
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED) # 10-bit Bayer Pattern
        
        #print('max pel value -- ', np.max(raw))
        ch_R  = raw[0::2, 0::2]
        ch_Gb = raw[0::2, 1::2]
        ch_Gr = raw[1::2, 0::2]
        ch_B  = raw[1::2, 1::2]
    
        tr_sample = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
                
        lab_path = os.path.join(lab_path, os.path.basename(path))[:-3]+'jpg'
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
        
        tr_sample = np.reshape(tr_sample, (1,)+np.shape(tr_sample))
        lab_sample = np.reshape(lab_sample, (1,)+np.shape(lab_sample))
        
        return (tr_sample, lab_sample)
    
    def next_batch(self):
    
        while(1):
            
            if len(self.buffer) == 0:
                time.sleep(0.1)
                
            else:
                self.lock.acquire()
                item = self.buffer.pop(0)
                self.lock.release()
                #print('Retrieved - Buffer Size  {:d}'.format(len(bbuffer)))
                break
    
        return item
    
    def close(self):
        
        self.end_flag[0] = True
        
        
    

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
            
        self.image_data = np.moveaxis(np.reshape(images, (-1, 3, 32, 32)), 1, -1)
        self.label_data = np.reshape(labels, (-1,))
        self.image_size = len(self.image_data)
        self.label_size = len(self.label_data)
        print('image data size - ', self.image_size)
        test = self._unpickle(os.path.join(data['data_path'], data['test_name']))
        self.eval_data = np.moveaxis(np.reshape(test['data'], (-1, 3, 32, 32)), 1, -1)
        self.eval_label = np.reshape(test['labels'], (-1))
        self.eval_size = FLAGS.eval_size
        self.eval_num = len(self.eval_data)//FLAGS.eval_size
        self.eval_pos = 0
        print('evaluation data size - ', len(self.eval_data))
        
        self.batch_size = FLAGS.batch_size
        self.index = np.arange(self.image_size)
        self._shuffle()
        
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
        images = []
        labels = []
        
        for _ in range(self.batch_size):
            
            #image = np.moveaxis(np.reshape(self.image_data[self.index[self.position]], (3, -1)), 0, -1)
            #images.append(np.reshape(image, (32, 32, 3)))
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
