import numpy as np
import time
from multiprocessing import shared_memory
import batch_reader


class IpcShm:
    def __init__(self, cfg):
        self.shm_name = cfg.shm_name
        self.mem = np.zeros(cfg.shm_shape, dtype=cfg.shm_dtype)
        self.shm = None
        self.attached_counter = None
        self.attached_array = None
        self.reader = batch_reader.BatchReader(cfg)

    def get_ipc_info(self):
        if self.attached_counter is None:
            counter = 0
        else:
            counter = self.attached_counter[0]
        # return f'Current Shared memory counter {counter}'
        return counter

    # producer
    def start_ipc(self):
        try:
            shm = shared_memory.SharedMemory(name=self.shm_name)
        except:
            # create a shared memory block of the same size as arr
            # [0, array] - 0 is a counter(int8) which indicates the length of the data
            shm = shared_memory.SharedMemory(create=True, size=self.mem.nbytes + 1, name=self.shm_name)
        # attach to the shared memory block
        attached_counter = shm.buf
        attached_array = np.ndarray(self.mem.shape, dtype=self.mem.dtype, buffer=shm.buf, offset=1)
        # print(f"\t\tcs... shm addr: {id(attached_array[0])}")

        while True:
            batch = self.reader.get_batch()
            attached_array[attached_counter[0]] = batch
            # self.lock.acquire()
            attached_counter[0] += 1
            # self.lock.release()
            # print(f'produced array! counter{attached_counter[0]} {attached_array[0][0][0][0]}')
            time.sleep(0)

        # finally before shm.unlike() to free and release shared memory block
        shm.close()

    # consumer
    def get_next(self):
        while self.shm is None:
            try:
                self.shm = shared_memory.SharedMemory(name=self.shm_name)
                # attach to the existing shared memory block
                self.attached_counter = self.shm.buf
                self.attached_array = np.ndarray(self.mem.shape, dtype=self.mem.dtype, buffer=self.shm.buf, offset=1)
            except:
                time.sleep(0)
        # print(f"ipc_shm memory addr: {id(self.attached_array[0])}")

        while True:
            # self.lock.acquire()
            counter = self.attached_counter[0]
            # decrement shared mem counter if it is read
            if counter != 0:
                batch = self.attached_array[counter - 1]
                self.attached_counter[0] -= 1
                break
            # self.lock.release()
            time.sleep(0)
        # print("\t\tconsuming...")
        return batch

    def close(self):
        self.shm.close()
        self.shm.unlink()