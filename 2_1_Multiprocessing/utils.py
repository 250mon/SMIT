# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import random
import time
import cv2
from multiprocessing import Process, Manager, Lock
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from operator import methodcaller

class Config(object):
    def __init__(self):
        # Setting dataset directory
        # self.DATA_DIR = '../Datasets/DIV2K/TRAIN'
        self.config_param = self.read_config()
        self.DATA_DIR = self.config_param['data_dir']
        assert os.path.exists(self.DATA_DIR)
        self.BUFFER_SIZE = 128  # fix this value (don't change)
        self.POOL_SIZE = 8  # number of Process(Thread) pool
        self.BATCH_SIZE = 4
        self.READ_SIZE = [512, 512]

    def read_config(self):
        with open('config', 'r') as fd:
            lines = fd.readlines()
        lines = map(methodcaller('strip'), lines)
        lines = list(map(methodcaller('split', ";"), lines))
        res_dict = {lines[i][0]: lines[i][1] for i in range(len(lines))}
        return res_dict

def init_pool(lock):
    global g_lock
    g_lock = lock

class ImageReader(object):
    def __init__(self, cfg):
        self.buffer = Manager().list([])
        self.buffer_size = cfg.BUFFER_SIZE
        self.lock = Lock()
        self.lock = Manager().Lock()

        self.img_list = glob.glob(os.path.join(cfg.DATA_DIR, "*.*"))
        random.shuffle(self.img_list)
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0

        self.pool_size = cfg.POOL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.read_size = np.array(cfg.READ_SIZE)

        # # single process
        # self.p = Process(target=self._start_buffer)
        # self.p.daemon = True
        # self.p.start()

        # # mp pool
        # # pool = Pool(self.pool_size, initializer=init_pool, initargs=(self.lock,))
        # pool = Pool(self.pool_size)
        # pool.apply_async(self._start_buffer)

        # thread pool
        pool = ThreadPool(self.pool_size)
        pool.apply_async(self._start_buffer)

        time.sleep(1)

    def _start_buffer(self):
        while True:
            batch = self._get_batch()
            while True:
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    time.sleep(0)

            self.lock.acquire()
            self.buffer.append(batch)
            self.lock.release()

            # g_lock.acquire()
            # self.buffer.append(batch)
            # g_lock.release()

    def _get_batch(self):
        batch = []
        for idx in range(self.batch_size):
            image = self._read_image(self.img_list[self.img_list_pos])
            batch.append(image)
            self.img_list_pos += 1
            if self.img_list_pos == self.img_list_size:
                self.img_list_pos = 0
                random.shuffle(self.img_list)
        return np.stack(batch, axis=0)

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # RANDOM CROP
        start = ((np.shape(img)[:2] - self.read_size) * np.random.uniform(size=2)).astype(np.int32)
        end = start + self.read_size
        crop = img[start[0]:end[0], start[1]:end[1], :]

        # AUGMENTATION - RANDOM ROTATE AND FLIPPING
        token = np.random.uniform()
        if token < 0.25:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        elif token < 0.5:
            crop = cv2.rotate(crop, cv2.ROTATE_180)
        elif token < 0.75:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

        token = np.random.uniform()
        if token < 0.5:
            crop = cv2.flip(crop, 1)  # flipcode > 0 LR flip

        return crop

    def get_next(self):
        while (1):
            if len(self.buffer):
                self.lock.acquire()
                item = self.buffer.pop(0)
                self.lock.release()
                break
            else:
                time.sleep(0)
        return item
