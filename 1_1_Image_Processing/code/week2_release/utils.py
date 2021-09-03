# -*- coding: utf-8 -*-
"""
Created on Wed Sept 4 16:05:47 2019

@author: Angelo
"""

import numpy as np
import os, glob, cv2




class ImageProcessing(object):
    
    def __init__(self, args):
        
        image_list = glob.glob(os.path.join(args.img_dir, "*.*"))
        self.img_list = self._eliminate_folders(image_list)
        self.img_list_size = len(self.img_list)
        self.img_list_pos = 0
        
        self.end_flag = False
        
    def _eliminate_folders(self, image_list):
        
        for idx in range(len(image_list)):
            if os.path.isdir(image_list[idx]):
                _ = image_list.pop(idx)
                
        return image_list
    
    def get_one_image(self):
        
        image = cv2.imread(self.img_list[self.img_list_pos])
        
        self.img_list_pos += 1
        if self.img_list_pos == self.img_list_size:
            self.img_list_pos = 0
            
        return image
    
    def cvtYCrCb(self, image):
        
        YCbCr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        return YCbCr
            
                        
    