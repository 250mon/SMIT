# -*- coding: utf-8 -*-
"""
Created on Mon Nov. 22 13:17:50 2021

@author: Angelo
"""
import os, argparse, cv2, time
import numpy as np
from multiprocessing import Process, Manager, Lock
from Vid_utils import Config, VideoManager


def get_arguments():
    parser = argparse.ArgumentParser(description="Implementation for Testing Video Player")
    
    parser.add_argument('--use_cam', 
                        type=bool, 
                        default= False,
                        help='cam or file', 
                        required = False)
    
    parser.add_argument('--video_in', 
                        type=str, 
                        default='Test1.mp4', 
                        help='The directory where the object images were located', 
                        required = False)
    
    
    return parser.parse_args()


def main():
    
    args = get_arguments()
    cfg = Config(args)
    
    first = True
    item_prev = None
    vid_prev = None
    
    end_flag = Manager().list([False])
    
    vid_mng = VideoManager(cfg, end_flag)
        
    vid_buffer = vid_mng.buffer
    vid_lock = vid_mng.lock
    
    while not len(vid_mng.img_property):
        time.sleep(0) # wait for video manager started
    
    vid_prop = vid_mng.img_property[0]
    
    print("Playing Video of size - ({:d}, {:d}) at {:.3f} FPS".format(vid_prop[0], vid_prop[1], vid_prop[2]))
    
    cv2.namedWindow('Images', cv2.WINDOW_AUTOSIZE)
    
    while(1):
        #start_time = time.time()
        while(1):
            
            if len(vid_buffer):
                vid_lock.acquire()
                image = vid_buffer.pop(0)
                vid_lock.release()
                break
            elif end_flag[0]:
                break
            else:
                time.sleep(0)
        
        if end_flag[0] and not len(vid_buffer):
            #print("Last Frame has been Reached !!!")
            break
        
        cv2.imshow('Images', image)
        
        key = cv2.waitKey(5) # 5ms display
        
        if key == ord('q'):
           break
        
        
        


if __name__ == '__main__':
    main()