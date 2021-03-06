from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, cv2, glob, time, random
import numpy as np
from multiprocessing import Process, Manager, Lock

class Config(object):
    
    label_color = [[128, 64, 128], [244, 35, 232], [70, 70, 70]
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
    
    cityscape_data={
            'train_img_path': '..\\cityscapes\\leftImg8bit\\train',
            'train_label_path': '..\\cityscapes\\gtFine\\train',
            'class_num': 19,
            'label_color': label_color
                }
    
    #augmentation
    augmentation = True
    resize_low = 0.5
    resize_high = 2.0
    
    weight_decay = 0.0005
    
    num_epoch = 400
    batch_size = 4
    learning_rate = 0.001
    
    BUFFER_SIZE = 16
    TRAIN_SIZE = (720, 720)
    
            
    def __init__(self, args):
        print('Setup configurations...')
        
        self.ckpt_dir = args.ckpt_dir
        self.log_dir = args.log_dir
        self.res_dir = args.res_dir
        

class CityscapesReader(object):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.label_colors = np.append(self.cfg.label_color, [[0, 0, 0]], axis=0)
        
        self.buffer = Manager().list([])
        self.buffer_size = cfg.BUFFER_SIZE
        self.lock = Lock()
        self.end_flag = Manager().list([False])
        
        self.batch_size = cfg.batch_size
        
        self.img_list = self._get_list()
        
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0
                                        
        self.p = Process(target=self._start_buffer)
        self.p.daemon=True
        self.p.start()
        time.sleep(0.5)
    
    
    def _get_list(self):
        
        train_cities = glob.glob(os.path.join(self.cfg.cityscape_data['train_img_path'], '*'))
        label_cities = glob.glob(os.path.join(self.cfg.cityscape_data['train_label_path'], '*'))

        train_cities.sort()
        label_cities.sort()
        
        img_list = []

        for idc, city in enumerate(train_cities):
            pngs = glob.glob(os.path.join(city, '*.png'))
                
            for idf, file in enumerate(pngs):
                fname = os.path.basename(file).split('_')
                fname[-1] = 'gtFine'
                fname += ['labelTrainIds.png']
                fname = '_'.join(fname)
                
                fname = os.path.join(label_cities[idc], fname)
                
                if os.path.exists(fname):
                    img_list.append((file, fname))
                else:
                    sys.exit('No matched - ', fname)

        print('List of (train_image, train_label) of ', len(img_list), '.... processing')                    
        return img_list
    
    
    def _start_buffer(self):
        
        while(1):
            
            if self.end_flag[0]:
                break
            
            _batch = self._get_batch()
                        
            while(1):
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    if self.end_flag[0]:
                        break
                    time.sleep(0.1)
                    
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
            tr, lab = self._read_image(self.img_list[self.img_list_pos], augment=self.cfg.augmentation)
            
            tr_cache.append(tr)
            lab_cache.append(lab)
            
            self.img_list_pos += 1
                
        tr_batch = np.stack(tr_cache, axis=0)
        lab_batch = np.stack(lab_cache, axis=0)
       
        return (tr_batch, lab_batch)
    
    def _read_image(self, path, augment=True):
        
        img_sample = cv2.imread(path[0], cv2.IMREAD_UNCHANGED)
        lab_sample = cv2.imread(path[1], cv2.IMREAD_UNCHANGED)
                
        tr_size = np.array(self.cfg.TRAIN_SIZE)
                
        if augment:
            # image augmentation - horizontal flip, and resizing
            
            flip = np.random.randint(1, 100) % 2
            if flip:
                img_sample = np.fliplr(img_sample)
                lab_sample = np.fliplr(lab_sample)
            
            resize_low = np.maximum(np.max(np.array(self.cfg.TRAIN_SIZE)/np.shape(img_sample)[:2]), self.cfg.resize_low)
            print('RESIZE LOW is - ', resize_low)
            ratio = np.random.uniform(low=resize_low, high=self.cfg.resize_high)
            new_size = np.flip(np.maximum(np.round(np.shape(img_sample)[:2] * np.array(ratio, dtype=np.float32)), tr_size)).astype(np.int32)
                        
            img_sample = cv2.resize(img_sample, tuple(new_size), interpolation=cv2.INTER_LANCZOS4)
            lab_sample = cv2.resize(lab_sample, tuple(new_size), interpolation=cv2.INTER_NEAREST)
                        
            crop_pos = np.round((np.shape(img_sample)[0:2] - tr_size)*np.random.uniform()).astype(np.int32)
            
            if len(np.shape(lab_sample)) == 2:
                lab_sample = np.expand_dims(lab_sample, axis=2)
                
            img_sample = img_sample[crop_pos[0]:crop_pos[0]+tr_size[0], crop_pos[1]:crop_pos[1]+tr_size[1], :]
            lab_sample = lab_sample[crop_pos[0]:crop_pos[0]+tr_size[0], crop_pos[1]:crop_pos[1]+tr_size[1], :]
            
        else:
            crop_pos = np.round((np.shape(img_sample) - tr_size)*np.random.uniform()).astype(np.int32)
            
            if len(np.shape(lab_sample)) == 2:
                lab_sample = np.expand_dims(lab_sample, axis=2)
            
            img_sample = img_sample[crop_pos[0]:crop_pos[0]+tr_size[0], crop_pos[1]:crop_pos[1]+tr_size[1], :]
            lab_sample = lab_sample[crop_pos[0]:crop_pos[0]+tr_size[0], crop_pos[1]:crop_pos[1]+tr_size[1], :]
                            
        return (img_sample, lab_sample)
    
    
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
        print('Closing Processes....................................')
        time.sleep(1)
        
        
    def show_cityscapes(self, images, labels):
    
        sq_row = int(np.sqrt(np.shape(images)[0]))
            
        total_image = []
        total_label = []
                
        for row in range(sq_row):
            row_img = [images[id + row*sq_row] for id in range(sq_row)]
            row_lab = [labels[id + row*sq_row] for id in range(sq_row)]
            
            total_image.append(np.concatenate(row_img, axis=1))
            total_label.append(np.concatenate(row_lab, axis=1))
            
        show_img = np.concatenate(total_image, axis=0)
        show_lab = np.concatenate(total_label, axis=0)
        
        h, w = np.shape(show_lab)[:2]
        
        show_lab = np.where(show_lab==255, 19, show_lab)
        num_classes = 20
        
        index = np.reshape(show_lab, (-1,))
        
        one_hot = np.eye(num_classes)[index]
        show_lab = np.reshape(np.matmul(one_hot, self.label_colors), (h, w, 3))
        
        cv2.imshow('Training Image', show_img)
        cv2.imshow('Label Image', show_lab.astype(np.uint8))
        key = cv2.waitKey(0)
        
        return key
        
    
