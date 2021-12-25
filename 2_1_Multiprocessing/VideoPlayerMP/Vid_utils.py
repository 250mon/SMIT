# -*- coding: utf-8 -*-
"""
Created on Wed Sept 4 16:05:47 2019

@author: Angelo
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0(all logs)', '1(filter out INFO)', '2(additionaly filter out WARNING', '3(additionally filter out ERROR'}

from multiprocessing import Process, Manager, Lock
import numpy as np
import time, cv2


class Config(object):
    
    # capture device with buffer parameters
    CAP_WIDTH = 1280
    CAP_HEIGHT = 720
    CAP_FPS = 24.
    
    CAP_NATIVE = 1 # use the file as it is
    
    BUFFER_SIZE = 1
        
        
    def __init__(self, args):
        print('Setup configurations........................................Done')
                
        self.use_cam = args.use_cam
        self.loc_vidFile = args.video_in
        

class VideoManager(object):
    
    def __init__(self, cfg, last_flag):
        
        self.cfg = cfg
        self.size_conv = 0
        
        self.buffer = Manager().list([])
        self.buffer_size = cfg.BUFFER_SIZE
        self.lock = Lock()
        
        self.last_flag = last_flag
        
        self.img_property = Manager().list([])
        
        self.p = Process(target=self._start_buffer)
        self.p.daemon=True
        self.p.start()
        
        print('Video Manager Process........................................Started')
        
        
    def _start_buffer(self):
        
        start_flag = True
        acc_frames = 0
        skipped_frames = 0
        success_grab = False
        
        if self.cfg.use_cam:
            cap = cv2.VideoCapture(0)
            if cap.isOpened() == False:
                raise SystemError('Capture Device Error - Not Opend')
                
            self._set_param(cap)
            self.img_property.append((self.cfg.CAP_WIDTH, self.cfg.CAP_HEIGHT, self.cfg.CAP_FPS))
            interval = 1./cap.get(5)
            time.sleep(0.1)
        else:
            cap = cv2.VideoCapture(self.cfg.loc_vidFile)
            if cap.isOpened() == False:
                raise ValueError('Video File Error - {:s} Not Opend'.format(self.cfg.loc_vidFile))
            
            self.img_property.append((int(cap.get(3)), int(cap.get(4)), cap.get(5)))
            # (width, height, fps)
            interval = 1./cap.get(5)
            
            if self.cfg.CAP_NATIVE:
                self.size_conv = 0
            elif (self.cfg.CAP_WIDTH, self.cfg.CAP_HEIGHT) != self.img_property[0][0:-1]:
                self.size_conv = 1
        
        while(1):
            
            #grab and retrieve
            while(1):
                if cap.grab():
                    ret, image = cap.retrieve()
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if self.size_conv == 1:
                        image = cv2.resize(image, (self.cfg.CAP_WIDTH, self.cfg.CAP_HEIGHT), interpolation = cv2.INTER_AREA)
                    acc_frames += 1
                    prev_acc = acc_frames
                    last_success = time.time()
                    
                    success_grab = True
                    
                    if start_flag:
                        start_time = last_success
                        start_flag = False
                    break
                else:
                    if (time.time() - last_success) > 1.0:
                        #rewind video file
                        if self.cfg.use_cam:
                            self.last_flag[0] = True
                        else:
                            #cap.set(1, 0) #rewind file
                            self.last_flag[0] = True
                    
                    if self.last_flag[0] == True:
                        break
                    time.sleep(0)
            
            if self.last_flag[0] == True:
                break
                        
            #buffer inspection
            while(1):
                if len(self.buffer) < self.buffer_size:
                    break
                else:
                    time.sleep(0)
            
            # fill buffer and sleep
            if success_grab:
                self.lock.acquire()
                self.buffer.append(image)
                self.lock.release()
                success_grab = False
            
            num_frames = (time.time() - start_time)/interval
            int_num_frames = int(num_frames)
            
            if int_num_frames > acc_frames:
                acc_frames = int_num_frames
                if acc_frames != prev_acc:
                    skipped_frames += (acc_frames - prev_acc)
                sleep_time = 0
            else:
                sleep_time = (1. - (num_frames - float(int_num_frames))) * interval
                
            prev_acc = acc_frames
            
            #print("retrieved frames of {:d} with sleep time {:.3f}\t skipped_frames :: {:d}".format(acc_frames, sleep_time, skipped_frames))
            time.sleep(sleep_time)
            
    def _set_param(self, cap):
        
        cap.set(3, self.cfg.CAP_WIDTH)
        cap.set(4, self.cfg.CAP_HEIGHT)
        cap.set(5, self.cfg.CAP_FPS)
        time.sleep(0.1)
        '''
        0.  CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
        1.  CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
        2.  CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
        3.  CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
        4.  CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
        5.  CV_CAP_PROP_FPS Frame rate.
        6.  CV_CAP_PROP_FOURCC 4-character code of codec.
        7.  CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
        8.  CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
        9.  CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
        10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
        11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
        12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
        13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
        14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
        15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
        16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
        17. CV_CAP_PROP_WHITE_BALANCE_U The U value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
        18. CV_CAP_PROP_WHITE_BALANCE_V The V value of the whitebalance setting (note: only supported by DC1394 v 2.x backend currently)
        19. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
        20. CV_CAP_PROP_ISO_SPEED The ISO speed of the camera (note: only supported by DC1394 v 2.x backend currently)
        21. CV_CAP_PROP_BUFFERSIZE Amount of frames stored in internal buffer memory (note: only supported by DC1394 v 2.x backend currently)
        '''

