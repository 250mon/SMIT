"""
Created on Sat Mar. 09 15:09:17 2019

@author: ygkim

main for mnist

"""

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 


import argparse, utils, time

def get_arguments():
    
    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2020')
    
        
    parser.add_argument('--ckpt_dir', 
                        type=str, 
                        default='./ckpt',
                        help='The directory where the checkpoint files are located', 
                        required = False)
    
    parser.add_argument('--log_dir', 
                        type=str, 
                        default='./logs',
                        help='The directory where the Training logs are located', 
                        required = False)
    
    parser.add_argument('--res_dir', 
                        type=str, 
                        default='./res',
                        help='The directory where the Training results are located', 
                        required = False)
        
    return parser.parse_args()

    
def main():
    
    args = get_arguments()
    cfg = utils.Config(args)
    
    print("---------------------------------------------------------")
    print("         Starting Cityscapes-Data Batch Processing Example")
    print("---------------------------------------------------------")
    
    cityscapes = utils.CityscapesReader(cfg)
            
    while 1:
        batch_image, batch_label = cityscapes.next_batch()
        
        
        key = cityscapes.show_cityscapes(batch_image, batch_label)
                
        if key == ord('q'):
            cityscapes.close()
            time.sleep(1.0)
            break

            
            
   
if __name__ == '__main__':
       
    main() 
