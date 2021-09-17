"""
Created on Sat Mar. 09 15:09:17 2019

@author: ygkim

main for mnist

"""

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 


import argparse, utils
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = None

def arg_process():
    
    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2019')
    
    parser.add_argument('--data_path', 
                        type=str, 
                        default='../data',
                        #default='.\\data',
                        help='The directory where the MNIST images were located', 
                        required = False)
    
    parser.add_argument('--img_name',
                        type=str, 
                        default='train-images-idx3-ubyte',
                        help='The file name for MNIST image', 
                        required = False)
    
    parser.add_argument('--label_name',
                        type=str, 
                        default='train-labels-idx1-ubyte',
                        help='The file name for MNIST labels', 
                        required = False)
    
    parser.add_argument('--batch_size',
                        type=int, 
                        default=256,
                        help='mini-batch size for training', 
                        required = False)
            
    args, unkowns = parser.parse_known_args()
    
    return args, unkowns

    
def main(_):
    
    print("---------------------------------------------------------")
    print("         Starting MNIST Batch Processing Example")
    print("---------------------------------------------------------")
    
    mnist_data = utils.mnist_data(FLAGS)
    
    while 1:
        batch_x, batch_y = mnist_data.next_batch()
        
        print("batch generation,   PRESS any key to proceed and 'q' to quit this program")
        
        key = utils.show_image(batch_x, batch_y)
        
        if key == ord('q'):
            break

            
            
   
if __name__ == '__main__':
    
    FLAGS, unparsed = arg_process()
    
    tf.app.run() 
