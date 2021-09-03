# -*- coding: utf-8 -*-
"""
Created on Mon Feb  17 13:17:50 2020

@author: Angelo
"""

import argparse, utils, model, time
import numpy as np
import tensorflow.compat.v1 as tf

def get_arguments():
    
    parser = argparse.ArgumentParser('Implementation for MNIST handwritten digits 2020')
    
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=400,
                        help='Parameter for learning rate', 
                        required = False)
    
    parser.add_argument('--ld_epoch', 
                        type=int, 
                        default=300,
                        help='Learning Rate Linear Decay Start Epoch', 
                        required = False)
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=100,
                        help='parameter for batch size', 
                        required = False)
    
    parser.add_argument('--eval_size', 
                        type=int, 
                        default=1000,
                        help='parameter for batch size', 
                        required = False)
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.0005, #VGG16
                        #default=0.0001, # ResNet34
                        help='Parameter for learning rate', 
                        required = False)
    
    parser.add_argument('--net_type', 
                        type=str, 
                        #default='Dense',
                        #default='Conv',
                        default='VGG16',
                        #default='ResNet34',
                        help='Parameter for Network Selection', 
                        required = False)
    
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
    
    
    return parser.parse_args()


def main():
    
    args = get_arguments()
    cfg = utils.DataPreprocesser(args)
    
    print("---------------------------------------------------------")
    print("          Starting CIFAR-10 Mini-Batch Training")
    print("---------------------------------------------------------")
    
    #mnist = utils.MnistReader(cfg)
    cifar10 = utils.Cifar10DataSets(cfg)
    lrctrl = utils.LRController(cfg)
    
    if cfg.net_type == 'Dense':
        net = model.DenseNet(cfg)
    elif cfg.net_type == 'Conv':
        net = model.ConvNet(cfg)
    elif cfg.net_type == 'VGG16':
        cfg.weight_decay = 0.0005
        net = model.VGG16(cfg)
    elif cfg.net_type == 'ResNet34':
        cfg.weight_decay = 0.0002
        net = model.ResNet34(cfg)
    else: 
        raise NotImplementedError('Network Type is Not Defined')
        
    _train_op, _wd_loss, _loss, _logits, _sum_vgg = net.optimizer()
    
    per_epoch = cifar10.image_size//cfg.batch_size
    st_time = time.time()
    
    use_drop = True
    
    total_accuracy = 0.
    max_accuracy = 0.
    
    st_epoch = net.start_epoch
    for epoch in range(st_epoch, cfg.num_epoch):
                        
        mean_cost = 0.
        mean_wd_cost = 0.
        curr_lr = lrctrl.get_lr(epoch)
    
        if epoch%2 == 0:
            bt_reset = True
            
        if epoch > 100:
            use_drop=False
                                
        for step in range(per_epoch):
            
            images, labels = cifar10.next_batch()
            feed_dict = {net.batch_img:images, net.batch_lab:labels, net.btrain:True, net.breset:bt_reset, net.buse_drop:use_drop, net.learning_rate:curr_lr}
            
            _, wd_loss, loss = net.sess.run((_train_op, _wd_loss, _loss), feed_dict=feed_dict)
            bt_reset = False
                        
            mean_cost += loss
            mean_wd_cost += wd_loss
                
        mean_cost /= float(per_epoch)
        mean_wd_cost /= float(per_epoch)
        
        print("Learning at {:d} epoch :: WD_Cost - {:1.8f}, Cost - {:1.8f}".format(epoch, mean_wd_cost, mean_cost))
        
        #################################################################
        # EVALUATION
        #################################################################
        if (epoch+1)%4 == 0:
            total_accuracy = 0.
            
                        
            for setp in range(cifar10.eval_num):
                eval_image, eval_label = cifar10.eval_batch()
                
                feed_dict = {net.batch_img:eval_image, net.batch_lab:eval_label, net.btrain:False, net.breset:False, net.buse_drop:False}
                logits = net.sess.run(_logits, feed_dict=feed_dict)
                
                correct_prediction = np.equal(np.argmax(logits, axis=1), eval_label)
                total_accuracy += np.mean(correct_prediction)
                
            total_accuracy /= cifar10.eval_num
            
            elapsed = time.time() - st_time
            emin = elapsed//60
            esec = elapsed - emin*60
            print("Evaluation at %d epoch " %epoch, "==== Accuracy of %1.4f" %total_accuracy, "(Current Max of %1.4f)" %max_accuracy, "  (elapsed - %d min. %1.2f sec.)"%(emin, esec))
        
        ###############################################################
        # Summary and Checkpoint
        ################################################################
        feed_dict = {net.sum_losses:(mean_wd_cost, mean_cost, total_accuracy)}
        summaries = net.sess.run(_sum_vgg, feed_dict=feed_dict)
        net.writer.add_summary(summaries, epoch)    
        
        if total_accuracy > max_accuracy:
            net.save(epoch)
            max_accuracy = total_accuracy

    
if __name__ == '__main__':
    main()
