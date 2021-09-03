# -*- coding: utf-8 -*-
"""
Created on Mon Feb  17 13:17:50 2020

@author: Angelo
"""

import argparse, utils, model
import numpy as np

def get_arguments():
    
    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2020')
    
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=50,
                        help='Parameter for learning rate', 
                        required = False)
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=256,
                        help='parameter for batch size', 
                        required = False)
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.0001,
                        help='Parameter for learning rate', 
                        required = False)
    
    parser.add_argument('--net_type', 
                        type=str, 
                        #default='Dense',
                        default='Conv',
                        help='Parameter for Network Selection', 
                        required = False)
    
    return parser.parse_args()


def main():
    
    args = get_arguments()
    cfg = utils.DataPreprocesser(args)
    
    print("---------------------------------------------------------")
    print("          Starting MNIST Mini-Batch Training")
    print("---------------------------------------------------------")
    
    mnist = utils.MnistReader(cfg)
    
    if cfg.net_type == 'Dense':
        net = model.DenseNet(cfg)
    elif cfg.net_type == 'Conv':
        net = model.ConvNet(cfg)
    else: 
        raise NotImplementedError('Network Type is Not Defined')
        
    _train_op, _loss, _logits = net.optimizer()
    
    per_epoch = mnist.image_size//cfg.batch_size
    for epoch in range(cfg.num_epoch):
        
        mean_cost = 0.
        
        for step in range(per_epoch):
            
            images, labels = mnist.next_batch()
            feed_dict = {net.batch_img:images, net.batch_lab:labels}
            
            _, loss = net.sess.run((_train_op, _loss), feed_dict=feed_dict)
            mean_cost += loss
                
        mean_cost /= float(per_epoch)
        print("Learning at %d epoch " %epoch, "==== Cost of %1.8f" %mean_cost)
        
        #################################################################
        # EVALUATION
        #################################################################
        feed_dict = {net.batch_img:mnist.eval_data[0], net.batch_lab:mnist.eval_data[1]}
        logits = net.sess.run(_logits, feed_dict=feed_dict)
        
        correct_prediction = np.equal(np.argmax(logits, axis=1), mnist.eval_data[1])
        accuracy = np.mean(correct_prediction)
        print("Evaluation at %d epoch " %epoch, "==== Accuracy of %1.4f" %accuracy)

    
if __name__ == '__main__':
    main()
