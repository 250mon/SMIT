# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import random
import time
import cv2
from multiprocessing import Process, Manager, Lock


class Config(object):
    def __init__(self, data_dir):
        # Setting dataset directory
        # self.DATA_DIR = '../Datasets/DIV2K/TRAIN'
        self.DATA_DIR = data_dir
        self.BUFFER_SIZE = 128  # fix this value (don't change)
        self.POOL_SIZE = 8  # number of Process(Thread) pool
        self.BATCH_SIZE = 4
        self.READ_SIZE = [512, 512]


class ImageReader(object):
    def __init__(self, cfg):
        self.buffer = Manager().list([])
        self.buffer_size = cfg.BUFFER_SIZE
        self.lock = Lock()

        self.img_list = glob.glob(os.path.join(cfg.DATA_DIR, "*.*"))
        random.shuffle(self.img_list)
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0

        self.pool_size = cfg.POOL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.read_size = np.array(cfg.READ_SIZE)

        self.p = Process(target=self._start_buffer)
        self.p.daemon = True
        self.p.start()
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
