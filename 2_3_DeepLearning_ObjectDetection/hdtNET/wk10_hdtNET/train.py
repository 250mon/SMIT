# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:17:50 2019

@author: Angelo
"""

import argparse
import time, copy

import numpy as np

from model import hdtNET
from utils import Config, ImageReader


def get_arguments():
    parser = argparse.ArgumentParser(description="Implementation for ICNet Semantic Segmentation 2019")
    
    parser.add_argument("--dataset", 
                        type=str,
                        default='coco',
                        help="Which dataset to trained with",
                        choices=['cityscapes', 'ade20k', 'coco', 'others'],
                        required=False)
    
    parser.add_argument("--use_aug", 
                        default=True,
                        help="Whether to use data augmentation for inputs during the training.",
                        required=False)
    
    parser.add_argument('--log_dir', 
                        type=str, 
                        default='./logs',
                        help='The directory where the Training logs are located', 
                        required = False)
    
    parser.add_argument('--ckpt_dir', 
                        type=str, 
                        default='./ckpt',
                        help='The directory where the checkpoint files are located', 
                        required = False)
    
    parser.add_argument('--res_dir', 
                        type=str, 
                        default='./res',
                        help='The directory where the result images are located', 
                        required = False)
    
    return parser.parse_args()


class TrainConfig(Config):
    def __init__(self, arguments):
        Config.__init__(self, arguments)
        
        self.mode = 'train'
        self.sigma = 1.5
'''        
class EvalConfig(Config):
    def __init__(self, arguments):
        Config.__init__(self, arguments)
        
        self.mode = 'eval'
        self.BATCH_SIZE = 1
'''
def ReplayBufferProcess(train_net, rep_buffer, buffer_size, keep_cond, epochs, _train_op, _losses):
    
    buff_id = 0
    Net = train_net
    
    while(1):
        sec_buffer = []
        sec_buffer_size = 0
        for steps in range(buffer_size):
            start_time = time.time()
            
            image, label = rep_buffer[steps]
    
            train_fd = {Net.g_step: epochs, Net.img_in: image, Net.gt_in: label}
            _, losses = Net.sess.run([_train_op, _losses], feed_dict=train_fd)
                        
            if losses[1] > keep_cond:
                    sec_buffer.append((image, label))
                    sec_buffer_size += 1
            
            duration = time.time() - start_time
            print('step {:d}/{:d} REPLAY-BUFFER({:d})\t total loss = {:.3f}, loss-1/16 = {:.3f}, loss-1/4 = {:.3f}, opt_loss: {:.3f} ({:.3f} sec/step) '.\
                        format(steps, buffer_size, buff_id, losses[2], losses[0], losses[1], losses[3], duration))
        
        buff_id += 1
        
        if sec_buffer_size == 0:
            break
        elif buff_id == 10:
            break
        else:
            buffer_size = sec_buffer_size
            rep_buffer = copy.deepcopy(sec_buffer)

def get_next(bbuffer, block):
    
    while(1):
            
        if len(bbuffer):
            block.acquire()
            item = bbuffer.pop(0)
            block.release()
            #print('Pop Out Buffered Item - remaining', len(bbuffer))
            break
            
        else:
            time.sleep(0)
            
    return item        


def main():
    
    args = get_arguments()

    train_cfg = TrainConfig(args)
    #eval_cfg = EvalConfig(args)

    # Setup training network and training samples
    Reader = ImageReader(train_cfg)
    time.sleep(0.5)
    batch_buffer = Reader.buffer
    batch_lock = Reader.lock
    #eval_reader = ImageReader(eval_cfg)
    
    Net = hdtNET(train_cfg)
    
    # get train operator, loss information and summaries
    _train_op, _losses, _summaries, _Preds = Net.optimizer()
    
    # Visualizer Generation
    #vis = Visualizer(eval_cfg)
    
    # Iterate over training steps.
    global_step = Net.start_step
    epoch_step = int(Reader.img_list_size/train_cfg.BATCH_SIZE + 0.5)
        
    start_epoch = int(global_step/epoch_step)
    SAVE_STEP = int(epoch_step * train_cfg.SAVE_PERIOD)
    
    keep_cond = float(10000.)
    #size = (train_cfg.BATCH_SIZE, train_cfg.TRAIN_SIZE[0], train_cfg.TRAIN_SIZE[1], 3)
    #img_buf = np.zeros(size, dtype=np.float32)
    #size = (train_cfg.BATCH_SIZE, train_cfg.TRAIN_SIZE[0], train_cfg.TRAIN_SIZE[1], 1)
    #lbl_buf = np.zeros(size, dtype=np.uint8)
    
    for epochs in range(start_epoch, train_cfg.TRAIN_EPOCHS):
        
        epoch_loss = None
        epoch_loss2 = None
        
        rep_buffer = []
        buffer_size = 0
        
        start_batch = global_step%epoch_step
        
        for steps in range(start_batch, epoch_step):
            start_time = time.time()
            
            image, label = get_next(batch_buffer, batch_lock)
    
            train_fd = {Net.g_step: epochs, Net.img_in: image, Net.gt_in: label}
            _, losses, Preds = Net.sess.run([_train_op, _losses, _Preds], feed_dict=train_fd)
            
            a_losses = np.array(losses)
            if epoch_loss is None:
                epoch_loss = a_losses
                epoch_loss2 = a_losses * a_losses
            else:
                epoch_loss += a_losses
                epoch_loss2 += (a_losses * a_losses)
                
            if a_losses[1] > keep_cond:
                rep_buffer.append((image, label))
                buffer_size += 1
            
            if global_step % SAVE_STEP == 0:
                Net.save(global_step)
                
            global_step += 1
            
            duration = time.time() - start_time
            print('step {:d}/{:d} ({:d}) \t total loss = {:.3f}, loss-1/16 = {:.3f}, loss-1/4 = {:.3f}, opt_loss: {:.3f} ({:.3f} sec/step)'.\
                        format(global_step, epoch_step*(epochs+1), epochs, losses[2], losses[0], losses[1], losses[3], duration))
        
        epoch_loss /= (epoch_step-start_batch) #mean
        epoch_loss2 /= (epoch_step-start_batch)
        epoch_loss2 -= (epoch_loss * epoch_loss) #variance
        
        keep_cond = epoch_loss[1] + train_cfg.sigma * np.sqrt(epoch_loss2[1])
        
        print('step {:d} AVERAGE\t total loss = {:.3f}, loss-1/16 = {:.3f}, loss-1/4 = {:.3f}, opt_loss = {:.3f}, keep_cond = {:.3f}, buffer_size = {:d} '.\
                        format(global_step, epoch_loss[2], epoch_loss[0], epoch_loss[1], epoch_loss[3], keep_cond, buffer_size))
        #################################################################################
        # SAVE IMAGE HERE
        ####################################################################################
        Reader.save_oneimage(image, label, Preds, epochs)
        
        ################################################################################
        # REPLAY BUFFER
        #################################################################################
        ReplayBufferProcess(Net, rep_buffer, buffer_size, keep_cond, epochs, _train_op, _losses)
        
        ############################################################################
        # PROCESS SUMMARY
        ###############################################################################
        
        feed_dict = {Net.sum_loss: epoch_loss}
        summaries = Net.sess.run(_summaries, feed_dict=feed_dict)
        Net.writer.add_summary(summaries, epochs)
        
        #Visualize Outputs
    
    
if __name__ == '__main__':
    main()