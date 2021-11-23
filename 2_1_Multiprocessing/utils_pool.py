# -*- coding: utf-8 -*-
import multiprocessing

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
        # IPC
        self.buffer = Manager().list([])
        self.buffer_size = cfg.BUFFER_SIZE
        # self.mp_q = multiprocessing.Queue()

        # Lock
        # self.lock = Lock()
        self.lock = Manager().Lock()

        self.img_list = glob.glob(os.path.join(cfg.DATA_DIR, "*.*"))
        random.shuffle(self.img_list)
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0

        self.pool_size = cfg.POOL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.read_size = np.array(cfg.READ_SIZE)

    def start_pool(self):
        # mp pool
        pool = Pool(self.pool_size)
        pool.starmap_async(self._start_buffer, [(self.buffer, i) for i in range(8)])

        # thread pool
        # pool = ThreadPool(self.pool_size)
        # pool.starmap_async(self._start_buffer, [(self.buffer, i) for i in range(8)])

        # # mp pool with queue
        # pool = Pool(self.pool_size)
        # pool.starmap_async(self._start_buffer, [(self.mp_q, i) for i in range(8)])

    def _start_buffer(self) #, buf, dummy):
        print("start buffer")
        buf = self.buffer
        pool = Pool(self.pool_size)
        while True:
            result = pool.starmap_async(self._get_batch)
            batch = result.get()
            # batch = self._get_batch()
            # print("got a batch")
            while True:
                # print(len(buf))
                if len(buf) < self.buffer_size:
                    break
                else:
                    time.sleep(0)
            # print("appending a batch")
            self.lock.acquire()
            buf.append(batch)
            self.lock.release()

            # g_lock.acquire()
            # self.buffer.append(batch)
            # g_lock.release()

    def get_next(self):
        while True:
            if len(self.buffer):
                self.lock.acquire()
                item = self.buffer.pop(0)
                self.lock.release()
                break
            else:
                time.sleep(0)
        return item

    def _start_queue(self, shared_q, dummy):
        print("start queue")
        # while True:
        #     batch = self._get_batch()
        #     shared_q.put(batch)

    def get_next_from_queue(self):
        return self.mp_q.get()

    def close_queue(self):
        self.mp_q.close()

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

