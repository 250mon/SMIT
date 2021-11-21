import cv2
import time
import numpy as np
import os
from utils import ImageReader, Config

# data_dir = '../Datasets/DIV2K/TRAIN'
data_dir = os.path.join('D:', 'Datasets', 'multiprocessing', '/DIV2K/TRAIN')

def main():
    
    start_time = time.time()
    
    cfg = Config()
    cfg.DATA_DIR = data_dir
    
    reader = ImageReader(cfg)
    
    loop = int(reader.img_list_size * 5) # five epoch simulation
    cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)
    
    for idx in range(loop):
                
        batch = reader.get_next()
        print('Buffer Fullness at ({:d}) - {:d}'.format(idx, len(reader.buffer)))
                        
        image = np.concatenate((np.concatenate((batch[0], batch[1]), axis=1), np.concatenate((batch[2], batch[3]), axis=1)), axis=0)
        cv2.imshow('Images', image)
        
        key = cv2.waitKey(5) # 5ms display
        
        if key == ord('q'):
           break
    
    duration = time.time() - start_time
    
    print('Multiprocessing Simulation - Done in {:.3f}(sec)'.format(duration))

    
                    
    
    
if __name__ == '__main__':
    main()