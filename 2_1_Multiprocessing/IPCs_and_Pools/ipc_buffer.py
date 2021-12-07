import time
from multiprocessing import Manager
import batch_reader


class IpcBuffer:
    def __init__(self, cfg):
        self.buffer_size = cfg.buffer_size
        if cfg.buffer is None:
            cfg.buffer, cfg.lock = self._create_buffer()
        self.buffer = cfg.buffer
        self.lock = cfg.lock
        self.reader = batch_reader.BatchReader(cfg)

    def _create_buffer(self):
        return Manager().list([]), Manager().Lock()

    def get_ipc_info(self):
        # return f'Current Buffer length {len(self.buffer)}'
        return len(self.buffer)

    # producer
    def start_ipc(self):
        while True:
            batch = self.reader.get_batch()
            while True:
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    time.sleep(0)
            self.lock.acquire()
            self.buffer.append(batch)
            self.lock.release()

    # consumer
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

    def close(self):
        return
