import numpy as np
import os
import glob
import random
import time
import cv2
from multiprocessing import Process, Manager, Lock
from multiprocessing import shared_memory
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from operator import methodcaller


class BatchReader:
    def __init__(self, cfg):
        self.batch_size = cfg.batch_size
        self.img_list = cfg.img_list
        self.img_list_size = cfg.img_list_size
        self.read_size = np.array(cfg.read_size)
        self.img_list_pos = 0

    def get_batch(self):
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
        # img dtype uint8
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

