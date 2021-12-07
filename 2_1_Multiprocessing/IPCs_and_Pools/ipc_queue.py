from multiprocessing import Manager
import batch_reader


class IpcQueue:
    def __init__(self, cfg):
        self.buffer_size = cfg.buffer_size
        if cfg.queue is None:
            cfg.queue, cfg.lock = self._create_queue()
        self.queue = cfg.queue
        self.lock = cfg.lock
        self.reader = batch_reader.BatchReader(cfg)

    def _create_queue(self):
        return Manager().Queue(), Manager().Lock()

    def get_ipc_info(self):
        # return f'Current Buffer length {self.queue.qsize()}'
        return self.queue.qsize()

    # producer
    def start_ipc(self):
        while True:
            batch = self.reader.get_batch()
            self.queue.put(batch)

    # consumer
    def get_next(self):
        return self.queue.get()

    def close(self):
        return
