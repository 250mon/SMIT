import cv2
import time
import numpy as np
from utils_pool import ImageReader, Config
import logging


def main(pool_type, pool_size, ipc_type):
    start_time = time.time()
    cfg = Config(pool_type=pool_type, pool_size=pool_size, ipc_type=ipc_type)
    reader = ImageReader(cfg)
    loop = int(cfg.img_list_size * 5)  # five epoch simulation
    cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)
    reader.start_reader()

    ipc_occupied = 0
    for idx in range(loop):
        batch = reader.get_next()
        # print(f'Epoch{idx}, ', reader.get_ipc_info())
        ipc_occupied += reader.get_ipc_info()

        image = np.concatenate(
            (np.concatenate((batch[0], batch[1]), axis=1), np.concatenate((batch[2], batch[3]), axis=1)), axis=0)

        cv2.imshow('Images', image)
        key = cv2.waitKey(5)  # 5ms display
        if key == ord('q'):
            break

    reader.close()

    duration = time.time() - start_time
    avg_ipc_occupied = 1.0 * ipc_occupied / loop
    # print(f'{ipc_type}_{pool_type}_{pool_size} {duration:.3f}s {avg_ipc_occupied:.2f}')
    output = f'{ipc_type}_{pool_type}_{pool_size} {duration:.3f}s {avg_ipc_occupied:.2f}'
    logging.basicConfig(filename='log', level=logging.INFO)
    logging.info(output)
    print(output)


if __name__ == '__main__':
    main(pool_type=None, pool_size=0, ipc_type='buffer')
    main(pool_type='mp_pool', pool_size=2, ipc_type='buffer')
    main(pool_type='mp_pool', pool_size=4, ipc_type='buffer')
    main(pool_type='mp_pool', pool_size=8, ipc_type='buffer')
    main(pool_type='mp_pool', pool_size=16, ipc_type='buffer')
    main(pool_type='mp_pool', pool_size=32, ipc_type='buffer')
    main(pool_type='thread_pool', pool_size=2, ipc_type='buffer')
    main(pool_type='thread_pool', pool_size=4, ipc_type='buffer')
    main(pool_type='thread_pool', pool_size=8, ipc_type='buffer')
    main(pool_type='thread_pool', pool_size=16, ipc_type='buffer')
    main(pool_type='thread_pool', pool_size=32, ipc_type='buffer')
    main(pool_type=None, pool_size=0, ipc_type='queue')
    main(pool_type='mp_pool', pool_size=2, ipc_type='queue')
    main(pool_type='mp_pool', pool_size=4, ipc_type='queue')
    main(pool_type='mp_pool', pool_size=8, ipc_type='queue')
    main(pool_type='mp_pool', pool_size=16, ipc_type='queue')
    main(pool_type='mp_pool', pool_size=32, ipc_type='queue')
    main(pool_type='thread_pool', pool_size=2, ipc_type='queue')
    main(pool_type='thread_pool', pool_size=4, ipc_type='queue')
    main(pool_type='thread_pool', pool_size=8, ipc_type='queue')
    main(pool_type='thread_pool', pool_size=16, ipc_type='queue')
    main(pool_type='thread_pool', pool_size=32, ipc_type='queue')
    main(pool_type=None, pool_size=0, ipc_type='shm')
    main(pool_type='mp_pool', pool_size=2, ipc_type='shm')
    main(pool_type='mp_pool', pool_size=4, ipc_type='shm')
    main(pool_type='mp_pool', pool_size=8, ipc_type='shm')
    main(pool_type='mp_pool', pool_size=16, ipc_type='shm')
    main(pool_type='mp_pool', pool_size=32, ipc_type='shm')
    main(pool_type='thread_pool', pool_size=2, ipc_type='shm')
    main(pool_type='thread_pool', pool_size=4, ipc_type='shm')
    main(pool_type='thread_pool', pool_size=8, ipc_type='shm')
    main(pool_type='thread_pool', pool_size=16, ipc_type='shm')
    main(pool_type='thread_pool', pool_size=32, ipc_type='shm')
