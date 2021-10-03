from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, cv2, glob, time, random
import numpy as np
from multiprocessing import Process, Manager, Lock


class CityscapesReader(object):
    def __init__(self, settings):
        self.label_color = [[128, 64, 128], [244, 35, 232], [70, 70, 70]
                       # 0 = road, 1 = sidewalk, 2 = building
            , [102, 102, 156], [190, 153, 153], [153, 153, 153]
                       # 3 = wall, 4 = fence, 5 = pole
            , [250, 170, 30], [220, 220, 0], [107, 142, 35]
                       # 6 = traffic light, 7 = traffic sign, 8 = vegetation
            , [152, 251, 152], [70, 130, 180], [220, 20, 60]
                       # 9 = terrain, 10 = sky, 11 = person
            , [255, 0, 0], [0, 0, 142], [0, 0, 70]
                       # 12 = rider, 13 = car, 14 = truck
            , [0, 60, 100], [0, 80, 100], [0, 0, 230]
                       # 15 = bus, 16 = train, 17 = motocycle
            , [119, 10, 32]]
        # 18 = bicycle

        # self.dataset_root = '/mnt/e/Datasets'
        # self.dataset_root = 'C:\\Users\\idiot\\sjy\\Datasets'
        self.dataset_root = '/home/ynjn/sdb/Datasets'
        # self.dataset_dir = 'cityscape-dist'
        self.dataset_dir = 'cityscape_subset'
        self.cityscape_data = {
            'train_img_path': os.path.join(self.dataset_root, self.dataset_dir, 'leftImg8bit', 'train'),
            'train_label_path': os.path.join(self.dataset_root, self.dataset_dir, 'gtFine', 'train'),
            'class_num': 19,
            'label_color': self.label_color
        }

        self.batch_size = settings.batch_size
        self.output_dir = settings.res_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.num_of_batches = 2975 // self.batch_size
        self.augmentation = True
        self.resize_low = 0.5
        self.resize_high = 2.0
        self.TRAIN_SIZE = (720, 720)

        self.label_colors = np.append(self.label_color, [[0, 0, 0]], axis=0)
        self.buffer = Manager().list([])
        self.buffer_size = 64
        self.lock = Lock()
        self.end_flag = Manager().list([False])
        
        self.img_list = self._get_list()
        
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0
                                        
        self.p = Process(target=self._start_buffer)
        self.p.daemon=True
        self.p.start()
        time.sleep(0.5)
    
    
    def _get_list(self):
        train_cities = glob.glob(os.path.join(self.cityscape_data['train_img_path'], '*'))
        label_cities = glob.glob(os.path.join(self.cityscape_data['train_label_path'], '*'))
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
            tr, lab = self._read_image(self.img_list[self.img_list_pos], augment=self.augmentation)
            
            tr_cache.append(tr)
            lab_cache.append(lab)
            
            self.img_list_pos += 1
                
        tr_batch = np.stack(tr_cache, axis=0)
        lab_batch = np.stack(lab_cache, axis=0)
        return tr_batch, lab_batch
    
    def _read_image(self, path, augment=True):
        img_sample = cv2.imread(path[0], cv2.IMREAD_UNCHANGED)
        lab_sample = cv2.imread(path[1], cv2.IMREAD_UNCHANGED)
        tr_size = np.array(self.TRAIN_SIZE)
        if augment:
            # image augmentation - horizontal flip, and resizing
            flip = np.random.randint(1, 100) % 2
            if flip:
                img_sample = np.fliplr(img_sample)
                lab_sample = np.fliplr(lab_sample)
            
            resize_low = np.maximum(np.max(np.array(self.TRAIN_SIZE)/np.shape(img_sample)[:2]), self.resize_low)
            #print('RESIZE LOW is - ', resize_low)
            ratio = np.random.uniform(low=resize_low, high=self.resize_high)
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
                            
        return img_sample, lab_sample
    
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

    # show 16 images into 4 x 4 formats
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

    # show only one image
    def show_cityscapes_ids(self, imgs, preds, gts, title, save=False):
        # process only the first image of the list
        # show_img = images[0]
        # show_lab = labels[0]

        # process only the first image of the list
        # sq_row = int(np.sqrt(np.shape(images)[0]))
        sq_row = 2
        total_imgs = []
        total_preds = []
        total_gts = []
        for row in range(sq_row):
            row_imgs = [imgs[id + row * sq_row] for id in range(sq_row)]
            row_preds = [preds[id + row * sq_row] for id in range(sq_row)]
            row_gts = [gts[id + row * sq_row] for id in range(sq_row)]

            total_imgs.append(np.concatenate(row_imgs, axis=1))
            total_preds.append(np.concatenate(row_preds, axis=1))
            total_gts.append(np.concatenate(row_gts, axis=1))

        total_imgs_concat = np.concatenate(total_imgs, axis=0)
        total_preds_concat = np.concatenate(total_preds, axis=0)
        total_gts_concat = np.concatenate(total_gts, axis=0)

        # convert img composed of id to colored img
        def idtocolor(img):
            h, w = np.shape(img)[:2]
            img = np.where(img == 255, 19, img)
            num_classes = 20
            index = np.reshape(img, (-1,))
            one_hot = np.eye(num_classes)[index]
            img = np.reshape(np.matmul(one_hot, self.label_colors), (h, w, 3))
            return img.astype(np.uint8)

        show_pred_color = idtocolor(total_preds_concat)
        show_gt_color = idtocolor(total_gts_concat)
        show_preds = np.cast(0.7 * total_imgs_concat + 0.3 * show_pred_color, np.uint8)
        show_gts = np.cast(0.7 * total_imgs_concat + 0.3 * show_gt_color, np.uint8)

        if save:
            filename = os.path.join(self.output_dir, title+'_pred.png')
            cv2.imwrite(filename, show_preds)
            filename = os.path.join(self.output_dir, title+'_gt.png')
            cv2.imwrite(filename, show_gts)
        else:
            cv2.imshow('Training Image', show_preds)
            cv2.imshow('Label Image', show_gt_color)
            key = cv2.waitKey(0)
            if key == ord('q'):
                time.sleep(1.0)
                exit(0)
