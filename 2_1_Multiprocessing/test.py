import cv2
import time
import numpy as np
import os
# from utils import ImageReader, Config
from utils_pool import ImageReader, Config


def main():
    start_time = time.time()
    cfg = Config()
    reader = ImageReader(cfg)
    loop = int(reader.img_list_size * 1)  # five epoch simulation
    # cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)
    # reader.start_pool()
    reader._start_buffer()

    for idx in range(loop):
        batch = reader.get_next()
        print('Buffer Fullness at ({:d}) - {:d}'.format(idx, len(reader.buffer)))

        # batch = reader.get_next_from_queue()
        # print('Queue Size at ({:d}) - {:d}'.format(idx, reader.mp_q.qsize()))

        image = np.concatenate(
            (np.concatenate((batch[0], batch[1]), axis=1), np.concatenate((batch[2], batch[3]), axis=1)), axis=0)

        # cv2.imshow('Images', image)
        # key = cv2.waitKey(5)  # 5ms display
        # if key == ord('q'):
        #     break

    reader.close_queue()

    duration = time.time() - start_time
    print('Multiprocessing Simulation - Done in {:.3f}(sec)'.format(duration))


if __name__ == '__main__':
    main()