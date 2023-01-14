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
        self.counter = 0
        self.rd_pos = 0

    def _create_buffer(self):
        return Manager().list([]), Manager().Lock()

    def get_ipc_info(self):
        # return f'Current Buffer length {len(self.buffer)}'
        return len(self.buffer)

    # producer
    def start_ipc(self):
        wt_pos = 0
        while True:
            batch = self.reader.get_batch()
            while True:
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    time.sleep(0)

            # # wrong; don't lock append, but counter
            # self.lock.acquire()
            # self.buffer.append(batch)
            # self.lock.release()

            for idx in range(len(batch)):
                self.buffer[wt_pos] = batch[idx]
                wt_pos = (wt_pos + 1) % self.buffer_size
                self.lock.acquire()
                self.counter += 1
                self.lock.release()
                while True:
                    if self.counter < self.buffer_size:
                        break
                    else:
                        time.sleep(0)

    # consumer
    def get_next(self):
        while True:
            if len(self.buffer):
                # # don't lock pop but counter
                # # pop(0) is time consuming job
                # self.lock.acquire()
                # item = self.buffer.pop(0)
                # self.lock.release()

                item = self.buffer[self.rd_pos]
                self.rd_pos = (self.rd_pos + 1) % self.buffer_size
                self.lock.acquire()
                self.counter -= 1
                self.lock.release()
                break
            else:
                time.sleep(0)
        return item

    def close(self):
        return
