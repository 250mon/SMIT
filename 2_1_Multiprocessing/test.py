import cv2
import time
import numpy as np
import os
# from utils import ImageReader, Config
from utils_pool2 import ImageReader, Config


def main():
    start_time = time.time()
    cfg = Config(pool_type=None, ipc_type='queue')
    # cfg = Config(pool_type='mp_pool', ipc_type='queue')
    # cfg = Config(pool_type=None, ipc_type='shm')
    # cfg = Config(pool_type='mp_pool', ipc_type='shm')
    reader = ImageReader(cfg)
    loop = int(reader.img_list_size * 1)  # five epoch simulation
    cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)
    reader.start_reader()

    for idx in range(loop):
        batch = reader.get_next()
        print(f'Epoch{idx}, ', reader.get_ipc_info())
        # print('Buffer Fullness at ({:d}) - {:d}'.format(idx, len(reader.buffer)))
        # print('Queue Size at ({:d}) - {:d}'.format(idx, reader.queue.qsize()))

        image = np.concatenate(
            (np.concatenate((batch[0], batch[1]), axis=1), np.concatenate((batch[2], batch[3]), axis=1)), axis=0)

        cv2.imshow('Images', image)
        # key = cv2.waitKey(5)  # 5ms display
        # if key == ord('q'):
        #     break

    reader.close()

    duration = time.time() - start_time
    print('Multiprocessing Simulation - Done in {:.3f}(sec)'.format(duration))


if __name__ == '__main__':
    main()