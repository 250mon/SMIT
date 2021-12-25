# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:05:47 2019

@author: Angelo
"""
import numpy as np

import os, glob, random, time, cv2
import PIL.Image as Image
import multiprocessing as mp
from multiprocessing import Process, Manager, Lock
from read_config import read_config

class Config(object):
    # Setting dataset directory
    # CITYSCAPES_DATA_DIR = '/home/ygkim/hdtNET/cityscapes'
    # COCO_DATA_DIR = '../Dataset/Train'
    dirs = read_config('wk10_config')
    CITYSCAPES_DATA_DIR = dirs['CITYSCAPES_DATA_DIR']
    COCO_DATA_DIR = dirs['COCO_DATA_DIR']

class Config(object):
    # Setting dataset directory
    CITYSCAPES_DATA_DIR = '/home/ygkim/hdtNET/cityscapes'
    COCO_DATA_DIR = '../Dataset/Train'
    COCO_EVAL_DIR = '../Dataset/Eval'
    #CITYSCAPES_DATA_DIR = 'D:\\hdtNET\\cityscapes'
    CITYSCAPES_train_list = os.path.join(CITYSCAPES_DATA_DIR, 'trainAttribute.txt')      
    CITYSCAPES_eval_list = os.path.join(CITYSCAPES_DATA_DIR, 'valAttribute.txt')
    '''
    label_colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[102, 102, 156], [190, 153, 153], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[250, 170, 30], [220, 220, 0], [107, 142, 35]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 251, 152], [70, 130, 180], [220, 20, 60]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 0, 0], [0, 0, 142], [0, 0, 70]
                # 12 = rider, 13 = car, 14 = truck
                ,[0, 60, 100], [0, 80, 100], [0, 0, 230]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[119, 10, 32]]
                # 18 = bicycle
    '''
    label_colors = [[0, 0, 142], [220, 20, 60]]
                    # 0 = else, 1 = person
                
    person_weight = 1.0
                
    # ImageNet RGB Mean
    IMG_MEAN = np.array((124.68, 116.779, 103.939), dtype=np.float32)
        
    cityscapes_param = {'name': 'cityscapes',
                    'num_classes': 19,
                    'ignore_label': 255,
                    'eval_size': [1024, 2048],
                    'eval_steps': 500,
                    'eval_list': CITYSCAPES_eval_list,
                    'train_list': CITYSCAPES_train_list,
                    'data_dir': CITYSCAPES_DATA_DIR,
                    'label_colors': label_colors}
    
    coco_param = {'name': 'coco-train2017',
                    'num_classes': 2,
                    'ignore_label': 100,
                    'data_dir': COCO_DATA_DIR,
                    'eval_dir': COCO_EVAL_DIR,
                    'label_colors': label_colors}
    
    ## add other dataset parameters 
    dataset_param = {'name': 'YOUR_OWN_DATASET',
                    'num_classes': 0,
                    'ignore_label': 0,
                    'eval_size': [0, 0],
                    'eval_steps': 0,
                    'eval_list': '/PATH/TO/YOUR_EVAL_LIST',
                    'train_list': '/PATH/TO/YOUR_TRAIN_LIST',
                    'data_dir': '/PATH/TO/YOUR_DATA_DIR'}
    
    visible_device = '0'

    ## You can modify following lines to train different training configurations.
    TRAIN_SIZE = [1280, 720] 
    #TRAIN_EPOCHS = 200 # num epochs for weight decay
    #DECAY_EPOCH = 20
    TRAIN_EPOCHS = 800 # num epochs for weight decay
    DECAY_EPOCH = 200
    SAVE_PERIOD = 2 # every N epochs
    
    # Weight Decay Param
    #WEIGHT_DECAY = 0.00004 #0.00005 in AlexNet, 0.0 for turn off
    WEIGHT_DECAY = 0.
    # Learning Rate Control Param
    LR_CONTROL = 'None' # [poly, linear, exponential, None]
    LEARNING_RATE = 0.0002
    POWER = 0.9
    MAX_ITERATION = 30000
    # Batch Normalization Learning Control
    BN_LEARN = True
    
   # Loss Function = LAMBDA1 * loss1 + LAMBDA4 * loss4 + LAMBDA16 * loss16 + weight-decay
    LAMBDA1 = 2.5
    LAMBDA4 = 1.0
    LAMBDA16 = 0.4
    
    # Dataset Processing Variables (Reader)
    BATCH_SIZE = 1
    BUFFER_SIZE = 2048 # number of batches for prefetch buffer
    #N_WORKERS = min(mp.cpu_count(), BATCH_SIZE)
    N_WORKERS = 8

    def __init__(self, args):
        print('Setup configurations...')
        
        if args.dataset == 'cityscapes':
            self.param = self.cityscapes_parm
        elif args.dataset == 'coco':
            self.param = self.coco_param
        else:
            raise NotImplementedError('NIY, except for citiscapes dataset')
        
        self.dataset = args.dataset
        
        self.ckpt_dir = args.ckpt_dir
        self.log_dir = args.log_dir
        self.res_dir = args.res_dir
        
    def display(self):
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                print("{:30} {}".format(a, getattr(self, a)))

            if a == ("param"):
                print(a)
                for k, v in getattr(self, a).items():
                    print("   {:27} {}".format(k, v))

        print("\n")
        


class ImageReader(object):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        
        self.buffer = Manager().list([])
        self.buffer_size = cfg.BUFFER_SIZE
        self.lock = Lock()
                
        self.batch_size = cfg.BATCH_SIZE
        
        self.img_list = glob.glob(os.path.join(cfg.param['data_dir'], "*.*"))
        random.shuffle(self.img_list)
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0
                        
        self.res_dir = cfg.res_dir
        self.label_colors = cfg.param['label_colors']
        self.num_classes = cfg.param['num_classes']
        
        self.aug_size = [0.5, 0.8, 1.2, 1.5, 2.0]
        
        self.p = Process(target=self._start_buffer)
        self.p.daemon=True
        self.p.start()
        time.sleep(0)
        
        self.eval_buf = self._get_eval(os.path.join(cfg.param['eval_dir'], "*.*"))
        
    def _get_eval(self, path):
        
        eval_list = glob.glob(path)
        eval_buf = []
        
        for idx in range(len(eval_list)):
            eval_buf.append(self._read_image((eval_list[idx], False)))
                        
            print('Batching EVALUATION DATA - {:d}'.format(idx))
            
        return eval_buf
        
        
    def _start_buffer(self):
                                
        self.Threads = mp.pool.ThreadPool(self.cfg.N_WORKERS)
        
        while(1):
            
            items = self._get_batch()
                                    
            while(1):
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    time.sleep(0)
                    
            self.lock.acquire()
            self.buffer += items
            self.lock.release()
            #print('Batched Type - ', type(tr_batch), type(gt_batch))
           
    def _get_batch(self):
        
        jobs = []
        for idx in range(self.cfg.N_WORKERS):
            jobs.append((self.img_list[self.img_list_pos], True))
            self.img_list_pos += 1
            
            if self.img_list_pos == self.img_list_size:
                self.img_list_pos = 0
                random.shuffle(self.img_list)
                
        image = self.Threads.map(self._read_image, jobs)
                                
        return image

    '''    
    def _get_batch(self):
        
        batch = []
        batch_size = 0
        
        data_size = self.img_list_size
                
        # batching data in cache
        for index in range(self.img_cache_pos, self.pool_size):
            
            if not len(batch):
                batch = np.array(self.img_cache[index])
                batch_size += 1
            else:
                batch = np.concatenate((batch, np.array(self.img_cache[index])), axis=0)
                batch_size += 1
        
        while(1):
            
            # read image using pool
            if self.img_list_pos + self.pool_size > data_size-1:
                self.img_list_pos = 0
                random.shuffle(self.img_list)
                #print('Shuffled - first item is ', self.img_list[0])
            
            until = self.img_list_pos + self.pool_size
            sub_data = self.img_list[self.img_list_pos:until]
            self.img_list_pos = until
            
            self.img_cache = []
            for index in range(len(sub_data)):
                self.img_cache.append(self._read_image(sub_data[index]))
                        
            self.img_cache_pos = 0
            
            to_go = self.batch_size - batch_size
            
            if to_go > self.pool_size:
                # save all
                for index in range(self.pool_size):
                    if not len(batch):
                        batch = np.expand_dims(self.img_cache[index], axis=0)
                        batch_size += 1
                    else:
                        new = np.expand_dims(self.img_cache[index], axis=0)
                        batch = np.concatenate((batch, new), axis=0)
                        batch_size += 1
            else:
                # save part and break
                for index in range(to_go):
                    if not len(batch):
                        batch = np.expand_dims(self.img_cache[index], axis=0)
                        batch_size += 1
                        self.img_cache_pos += 1
                    else:
                        new = np.reshape(self.img_cache[index], (1,)+np.shape(self.img_cache[index]))
                        batch = np.concatenate((batch, new), axis=0)
                        batch_size += 1
                        self.img_cache_pos += 1
                        
                break
                
        return batch
    '''
    def _read_image(self, arg):
        
        path, aug = arg
        
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if aug:
            # flipping Left/Right first and Up/Down then
            token = np.random.randint(2)
            if token:
                img = cv2.flip(img, 1)
                
            token = np.random.randint(2)
            if token:
                img = cv2.flip(img, 0)
            
            #rotate
            token = np.random.randint(4)
            if token == 1:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif token == 2:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif token == 3:
                img = cv2.rotate(img, cv2.ROTATE_180)
                
            #size augmentation (0.5~1.5)
            token = np.random.randint(6)
            
            if token != 5:
                new_size = (np.flip(np.shape(img)[:2])*self.aug_size[token]).astype(np.int32)
            
                img1 = cv2.resize(img[:,:,:3], new_size, interpolation=cv2.INTER_LANCZOS4)
                img2 = cv2.resize(img[:,:,3], new_size, interpolation=cv2.INTER_NEAREST)
            else:
                img1 = img[:,:,:3]
                img2 = img[:,:,3]
        else:
            img1 = img[:,:,:3]
            img2 = img[:,:,3]
        
        #img = np.concatenate((img1, np.expand_dims(img2, axis=2)), axis=2)
            
        try:
            return (np.expand_dims(img1, axis=0), np.expand_dims(np.expand_dims(img2, axis=2), axis=0))
        except:
            print(path)
    
    def save_oneimage(self, img, gt, pred, step):
        
        gt = gt//255
        
        h, w = np.shape(img)[1:3]
        one_hot = np.eye(self.num_classes)
        
        out_img = []
                
        pred_img = np.reshape(np.matmul(one_hot[np.reshape(pred, -1)], self.label_colors), (h, w, 3))
        pred_img = Image.fromarray(pred_img.astype(np.uint8))
        
        gt_img = np.reshape(np.matmul(one_hot[np.reshape(gt, -1)], self.label_colors), (h, w, 3))
        gt_img = Image.fromarray(gt_img.astype(np.uint8))
        
        src_img = Image.fromarray(img[0])
        g_img = np.array(Image.blend(src_img, gt_img, 0.5))
        
        col_img = np.concatenate((g_img, pred_img))
        #Image.fromarray(col_img.astype(np.uint8)).show()
        
        out_img = Image.fromarray(col_img.astype(np.uint8))
        #out_img.show()
        
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        
        filename = os.path.join(self.res_dir, "prd_{}.png".format(step))
        out_img.save(filename)
            
        return out_img
    
    def save_image(self, img, gt, pred, step):
        
        gt = gt//255
        
        dim = np.shape(img)
        pos = np.random.randint(0, dim[0]-4)
        h, w = dim[1:3]
        one_hot = np.eye(self.num_classes)
        
        out_img = []
                
        for index in range(pos, pos+4):
            
            pred_img = np.reshape(np.matmul(one_hot[np.reshape(pred[index], -1)], self.label_colors), (h, w, 3))
            pred_img = Image.fromarray(pred_img.astype(np.uint8))
            
            gt_img = np.reshape(np.matmul(one_hot[np.reshape(gt[index], -1)], self.label_colors), (h, w, 3))
            gt_img = Image.fromarray(gt_img.astype(np.uint8))
            
            src_img = Image.fromarray(img[index])
            
            #p_img = np.array(Image.blend(src_img, pred_img, 0.5))
            #p_img = Image.blend(src_img, pred_img, 0.3)
            
            g_img = np.array(Image.blend(src_img, gt_img, 0.5))
            #g_img = Image.blend(src_img, gt_img, 0.3)
            
            #col_img = np.concatenate((g_img, p_img))
            #col_img = np.concatenate((g_img, p_img))
            col_img = np.concatenate((g_img, pred_img))
            #Image.fromarray(col_img.astype(np.uint8)).show()
            
            if len(out_img) == 0:
                out_img = col_img
            else:
                out_img = np.concatenate((out_img, col_img), axis=1)
        
        out_img = Image.fromarray(out_img.astype(np.uint8))
        #out_img.show()
        
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)
        
        filename = os.path.join(self.res_dir, "prd_{}.png".format(step))
        out_img.save(filename)
            
        return out_img
    
