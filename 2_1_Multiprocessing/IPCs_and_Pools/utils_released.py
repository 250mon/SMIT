def get_next(reader):
    while (1):
        if reader.counter != 0:
            item = reader.buffer[reader.rd_pos]
            reader.rd_pos = (reader.rd_pos + 1) % reader.cfg.buffer_size
            reader.lock.acquire()
            reader.counter -= 1
            reader.lock.release()
            # print('Pop Out Buffered Item - remaining', len(bbuffer))
            break
        else:
            time.sleep(0)
    return item

def _start_buffer(self):
    self.Threads = mp.pool.ThreadPool(self.cfg.N_WORKERS)
    self.wt_pos = 0
    while (1):
        items = self._get_batch()
        while (1):
            if self.counter < self.buffer_size:
                break
            else:
                time.sleep(0)
        for idx in range(len(items))
            self.buffer[self.wt_pos] = items[idx]
            self.wt_pos = (self.wt_pos + 1) % self.buffer_size
            self.lock.acqure()
            self.counter += 1
            self.lock.release()
            while (1):
                if self.counter < self.buffer_size:
                    break
                else:
                    time.sleep(0)
        # print('Batched Type - ', type(tr_batch), type(gt_batch))

def _get_batch(self):
    jobs = []
    for idx in range(self.cfg.N_WORKERS):
        jobs.append((self.img_list[self.img_list_pos], True))
        self.img_list_pos += 1
        if self.img_list_pos == self.img_list_size:
            self.img_list_pos = 0
            random.shuffle(self.img_list)
    image = self.Threads.map(self._read_image, jobs)
    return image
