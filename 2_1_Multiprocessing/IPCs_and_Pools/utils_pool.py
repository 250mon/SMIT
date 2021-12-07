import numpy as np
import os
import glob
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from operator import methodcaller
import ipc_shm
import ipc_buffer
import ipc_queue


proc = None
pool = None
# pool_type = [None | 'mp_pool' | 'thread_pool']
# ipc_type = ['buffer' | 'queue']
class Config(object):
    def __init__(self, pool_type=None, pool_size=8, ipc_type='buffer'):
        self.config_param = self.read_config()
        self.DATA_DIR = self.config_param['data_dir']
        assert os.path.exists(self.DATA_DIR)
        self.buffer_size = 128  # fix this value (don't change)
        self.pool_size = pool_size  # number of Process(Thread) pool
        self.batch_size = 4
        self.read_size = [512, 512]

        self.ipc_type = ipc_type
        # reader
        self.img_list = glob.glob(os.path.join(self.DATA_DIR, "*.*"))
        self.img_list_size = len(self.img_list)
        # buffer
        self.buffer = None
        # queue
        self.queue = None
        # buffer or queue
        self.lock= None
        # shm
        self.shm_name = "shm_001"
        self.shm_shape = (self.buffer_size, *(4, 512, 512, 3))
        self.shm_dtype = np.uint8
        # pool
        self.pool_type = pool_type

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
            self.ipc = ipc_buffer.IpcBuffer(self.cfg)
        elif self.cfg.ipc_type == 'queue':
            self.ipc = ipc_queue.IpcQueue(self.cfg)
        else:
            self.ipc = ipc_shm.IpcShm(self.cfg)  # for the method get_next

    def get_ipc_info(self):
        return self.ipc.get_ipc_info()

    def get_next(self):
        return self.ipc.get_next()

    def start_reader(self):
        global proc

        if self.cfg.pool_type is None:
            # single child process
            proc = Process(target=self._start_ipc)
            # proc.daemon = True
            proc.start()
        else:
            # multiple child processes
            self._start_pool()

    def _start_pool(self):
        global pool

        if self.cfg.pool_type == 'mp_pool':
            pool = Pool(self.cfg.pool_size)
        else:
            pool = ThreadPool(self.cfg.pool_size)

        for i in range(8):
            pool.apply_async(self._start_ipc)

    def _start_ipc(self):
        start_ipc_func = {
            'buffer': self._start_buffer,
            'queue': self._start_queue,
            'shm': self._start_shm,
        }
        return start_ipc_func[self.cfg.ipc_type]()

    def _start_buffer(self):
        bufipc = ipc_buffer.IpcBuffer(self.cfg)
        bufipc.start_ipc()

    def _start_queue(self):
        qipc = ipc_queue.IpcQueue(self.cfg)
        qipc.start_ipc()

    def _start_shm(self):
        shmipc = ipc_shm.IpcShm(self.cfg)
        shmipc.start_ipc()

    def close(self):
        if self.cfg.pool_type == 'mp_pool':
            pool.terminate()
            pool.join()
        else:
            proc.terminate()
            proc.join()

        self.ipc.close()