# -*- coding: utf-8 -*-
"""
Created on Mon Feb  17 13:17:50 2020

@author: Angelo
"""

import argparse, utils, model
import numpy as np
import tensorflow.compat.v1 as tf

def get_arguments():
    
    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2020')
    
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=10,
                        help='Parameter for learning rate', 
                        required = False)
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256,
                        help='parameter for batch size', 
                        required = False)
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.001,
                        help='Parameter for learning rate', 
                        required = False)
    
    return parser.parse_args()


def main():
    
    args = get_arguments()
    cfg = utils.DataPreprocesser(args)
    
    print("---------------------------------------------------------")
    print("          Starting MNIST Mini-Batch Training")
    print("---------------------------------------------------------")
    
    mnist = utils.MnistReader(cfg)
    # TF forward pass network created by _build()
    net = model.DenseNet(cfg)
    # TF session created
    _train_op, _loss, _logits = net.optimizer()
    
    # number of repetition per epoch
    # for example, 60000 // 256 = 234
    per_epoch = mnist.image_size//cfg.batch_size
    
    # 10 epoch; the whole data was fed 10 times
    for epoch in range(cfg.num_epoch):
        
        mean_cost = 0.
        
        # For each epoch, 234 batches after which the whole data has been gotten through
        for step in range(per_epoch):
            
            images, labels = mnist.next_batch()
            feed_dict = {net.batch_img:images, net.batch_lab:labels}
            
            _, loss = net.sess.run((_train_op, _loss), feed_dict=feed_dict)
            mean_cost += loss
                
        mean_cost /= float(per_epoch)
        print("Learning at %d epoch " %epoch, "==== Cost of %1.4f" %mean_cost)

    
if __name__ == '__main__':
    main()
