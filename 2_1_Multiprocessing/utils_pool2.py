# -*- coding: utf-8 -*-
import multiprocessing

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


pool = None


def _start_shm(smname, smshape, smdtype):
    shm = shared_memory.SharedMemory(name=smname)
    # attach to the existing shared memory block
    attached_counter = shm.buf
    attached_array = np.ndarray(smshape, dtype=smdtype, buffer=shm.buf, offset=1)
    # attached_array = np.ndarray(reader.mem.shape, dtype=reader.mem.dtype, buffer=shm.buf, offset=1)
    # print(f"pd... shm: {id(attached_array[0])}")

    while True:
        batch = np.zeros((512, 512), dtype=np.int32)
        attached_array[attached_counter[0]] = batch
        attached_counter[0] += 1
        # print(f'produced array! {attached_array[0]}')
        print(f'produced array! {attached_counter[0]}')
        time.sleep(0)

    # finally before shm.unlike() to free and release shared memory block
    shm.close()


# pool_type = [None | 'mp_pool' | 'thread_pool']
# ipc_type = ['buffer' | 'queue']
class Config(object):
    def __init__(self, pool_type=None, ipc_type='buffer'):
        # Setting dataset directory
        # self.DATA_DIR = '../Datasets/DIV2K/TRAIN'
        self.config_param = self.read_config()
        self.DATA_DIR = self.config_param['data_dir']
        assert os.path.exists(self.DATA_DIR)
        self.BUFFER_SIZE = 128  # fix this value (don't change)
        self.POOL_SIZE = 8  # number of Process(Thread) pool
        self.BATCH_SIZE = 4
        self.READ_SIZE = [512, 512]

        self.pool_type = pool_type
        self.ipc_type = ipc_type

    def read_config(self):
        with open('config', 'r') as fd:
            lines = fd.readlines()
        lines = map(methodcaller('strip'), lines)
        lines = list(map(methodcaller('split', ";"), lines))
        res_dict = {lines[i][0]: lines[i][1] for i in range(len(lines))}
        return res_dict

class ImageReader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # IPC
        if self.cfg.ipc_type == 'buffer':
            self.ipc = Manager().list([])
            self.buffer_size = cfg.BUFFER_SIZE
            self.lock = Manager().Lock()
        elif self.cfg.ipc_type == 'queue':
            self.ipc = Manager().Queue()
            self.lock = Manager().Lock()
        else:
            # self.mem = np.zeros((self.cfg.BUFFER_SIZE, *(self.cfg.READ_SIZE)), dtype=np.int32)
            # # create a shared memory block of the same size as arr
            # # [0, array] - 0 is a counter(int8) which indicates the length of the data
            # self.shm = shared_memory.SharedMemory(create=True, size=self.mem.nbytes + 1)
            # # attach to the existing shared memory block
            # self.attached_counter = self.shm.buf
            # self.attached_array = np.ndarray(self.mem.shape, dtype=self.mem.dtype, buffer=self.shm.buf, offset=1)
            # # print(f"\t\tcs... shm addr: {id(attached_array[0])}")
            # shared memory name
            self.ipc = 'shm_001'
            self.mem = np.zeros((self.cfg.BUFFER_SIZE, *(4, 512, 512, 3)), dtype=np.int32)
            # self.lock = Manager().Lock()
            self.attach_flag = False
            self.shm = None

        self.img_list = glob.glob(os.path.join(cfg.DATA_DIR, "*.*"))
        random.shuffle(self.img_list)
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0

        self.pool_size = cfg.POOL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.read_size = np.array(cfg.READ_SIZE)

        self.start_ipc_func = {
            'buffer': self._start_buffer,
            'queue': self._start_queue,
            'shm': self._start_shm,
        }
        self.get_next_func = {
            'buffer': self._get_next_from_buffer,
            'queue': self._get_next_from_queue,
            'shm': self._get_next_from_shm,
        }

    def get_ipc_info(self):
        if self.cfg.ipc_type == 'buffer':
            return f'Current Buffer length {len(self.ipc)}'
        elif self.cfg.ipc_type == 'queue':
            return f'Current Queue size {self.ipc.qsize()}'
        else:
            return f'Current Shared memory counter {self.ipc}'

    def get_next(self):
        return self.get_next_func[self.cfg.ipc_type]()

    def start_reader(self):
        # if self.cfg.ipc_type == 'shm':
        #     # single child process
        #     proc = Process(target=_start_shm, args=(self.ipc, self.mem.shape, self.mem.dtype,))
        #     proc.start()
        if self.cfg.pool_type is None:
            # single child process
            proc = Process(target=self._start_ipc, args=(self.ipc,))
            # proc.daemon = True
            proc.start()
        else:
            # multiple child processes
            self._start_pool()

    def _start_ipc(self, ipc):
        # print('starting ipc')
        return self.start_ipc_func[self.cfg.ipc_type](ipc)

    def _start_buffer(self, buf):
        # print("start buffer")
        while True:
            batch = self._get_batch()
            while True:
                if len(buf) < self.buffer_size:
                    break
                else:
                    time.sleep(0)
            self.lock.acquire()
            buf.append(batch)
            self.lock.release()

    def _get_next_from_buffer(self):
        while True:
            if len(self.ipc):
                self.lock.acquire()
                item = self.ipc.pop(0)
                self.lock.release()
                break
            else:
                time.sleep(0)
        return item

    def _start_pool(self):
        # print("starting pool")
        global pool

        if self.cfg.pool_type == 'mp_pool':
            pool = Pool(self.pool_size)
        else:
            pool = ThreadPool(self.pool_size)

        for i in range(8):
            pool.apply_async(self._start_ipc, (self.ipc,))

    def _start_queue(self, shared_q):
        # print("start queue")
        while True:
            batch = self._get_batch()
            shared_q.put(batch)
            # sleep(0)

    def _get_next_from_queue(self):
        return self.ipc.get()

    # producer
    def _start_shm(self, smname):
        try:
            shm = shared_memory.SharedMemory(name=smname)
        except:
            # create a shared memory block of the same size as arr
            # [0, array] - 0 is a counter(int8) which indicates the length of the data
            shm = shared_memory.SharedMemory(create=True, size=self.mem.nbytes + 1, name=smname)
        # attach to the shared memory block
        attached_counter = shm.buf
        attached_array = np.ndarray(self.mem.shape, dtype=self.mem.dtype, buffer=shm.buf, offset=1)
        # print(f"\t\tcs... shm addr: {id(attached_array[0])}")

        while True:
            batch = self._get_batch()
            # self.lock.acquire()
            attached_array[attached_counter[0]] = batch
            attached_counter[0] += 1
            # self.lock.release()
            # print(f'produced array! {attached_array[0][0][0][0]}')
            time.sleep(0)

        # finally before shm.unlike() to free and release shared memory block
        shm.close()

    # consumer
    def _get_next_from_shm(self):
        while self.attach_flag is False:
            try:
                self.shm = shared_memory.SharedMemory(name=self.ipc)
            except:
                time.sleep(0)
            else:
                self.attach_flag = True

        # attach to the existing shared memory block
        attached_counter = self.shm.buf
        attached_array = np.ndarray(self.mem.shape, dtype=self.mem.dtype, buffer=self.shm.buf, offset=1)
        # print(f"pd... shm: {id(attached_array[0])}")

        while True:
            # self.lock.acquire()
            counter = attached_counter[0]
            # print(f"\tconsuming {counter}...")
            # decrement shared mem counter if it is read
            if counter != 0:
                attached_counter[0] -= 1
                break
            # self.lock.release()
            time.sleep(0)

        # print("\t\tconsuming...")
        return attached_array[0]
        # finally before shm.unlike() to free and release shared memory block
        # self.shm.close()

    def close(self):
        if self.cfg.pool_type == 'mp_pool':
            pool.terminate()
            pool.join()
        if self.cfg.ipc_type == 'shm':
            self.shm.close()
            self.shm.unlink()

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

