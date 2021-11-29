# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:58:28 2019

@author: Angelo
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow.compat.v1.bitwise as tw

import numpy as np

import os
from network import Network

    
class hdtNET(Network):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.num_classes = cfg.param['num_classes']
        
        self.loss_weight = (cfg.LAMBDA16, cfg.LAMBDA4, cfg.LAMBDA1)
        self.reservoir = {}
        self.losses = None  # = (loss_sub4, loss_sub2, loss_sub1, total_loss)
        
        self.sum_loss = tf.placeholder(dtype=tf.float32, shape=(2,))
        
        self.eps = tf.constant(1e-5)
        
        self.img_in = tf.placeholder(tf.uint8)
        self.img_size = tf.shape(self.img_in)[1:3]
        self.gt_in = tf.placeholder(tf.uint8)
        
        #self.ignore_label = cfg.param['ignore_label']
        self.ignore_label = tf.placeholder(tf.int32)
        
        self.start_pos = None # This is computed in _InputProcess
        self.image = self._InputProcess(self.img_in, multiples=16)
        self.label = tf.cast(tf.cast(self.gt_in, tf.int32)/255, tf.int32)
        
        self.image.set_shape([None, None, None, 3])
        self.label.set_shape([None, None, None, 1])
        
        
        self.batch_size = cfg.BATCH_SIZE
        self.person_weight = cfg.person_weight
        
        self.lr = cfg.LEARNING_RATE
        self.g_step = tf.placeholder(dtype=tf.float32, shape=())
        self.lr_width = float(cfg.TRAIN_EPOCHS - cfg.DECAY_EPOCH)
        self.decay_epoch = float(cfg.DECAY_EPOCH)
        
        #self.scale = 100000.
    
        super(hdtNET, self).__init__()
    
    def _InputProcess(self, image, multiples=16):
        
        dest = tf.cast(tf.ceil(self.img_size/multiples)*multiples, tf.int32)
        margin = dest - self.img_size
        
        h1 = tf.cast(margin[0]/2, tf.int32)
        h_pad = [h1, margin[0]-h1]
        
        w1 = tf.cast(margin[1]/2, tf.int32)
        w_pad = [w1, margin[1]-w1]
        
        pad_size = [[0, 0], h_pad, w_pad, [0,0]]
        
        self.start_pos = (h1, w1)
        
        return tf.cast(tf.pad(image, pad_size, 'REFLECT'), tf.float32)
    
    
    def _Encoder(self, enc_in, name=None, reuse=tf.AUTO_REUSE):
        
        #inputs = self._RGBtoYCbCr(tf.cast(enc_in, dtype=tf.float32, name='Enc_In'))
        #inputs = tf.cast(enc_in, dtype=tf.float32, name='Enc_In')/127.5 - 1.
        inputs = enc_in/127.5 - 1.
        
        with tf.variable_scope(name, reuse=reuse):
            
            (self.feed(inputs)
                 .space_to_depth(iblk_size=4, sname='enc_s2d1')
                 .convact(lfilter_shape=(3,3,48,48), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='enc_conv1')
                 .convact(lfilter_shape=(3,3,48,40), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='enc_conv2')
                 .to_reservoir(sname='enc_out') # H/4 x W/4
                 .convact(lfilter_shape=(3,3,40,64), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='fus_conv1')
                 .convact(lfilter_shape=(3,3,64,64), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='fus_conv2')
                 .to_reservoir(sname='fus_out')) # H/4 x W/4
                                                               
        return self.reservoir['enc_out']
    
    def _Transformer(self, tr_in, name=None, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse=reuse):
            (self.feed(tr_in)
                 .convact(lfilter_shape=(3,3,40,80), istride=2, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv1')
                 .convact(lfilter_shape=(3,3,80,80), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv2')
                 .convact(lfilter_shape=(3,3,80,160), istride=2, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv3')
                 .convact(lfilter_shape=(3,3,160,160), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv4')
                 .pypool(smethod='add', iCin=160, sname='pypool') # 'add' or 'concat'
                 .convact(lfilter_shape=(3,3,160,160), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv5')
                 #LOWEST LEVEL - UPSAMPLE # H/4 x W/4 pypool(self, tinputs, iCin, smethod='add', sname=None)
                 .deconvact(lfilter_shape=(4,4,128,160), istride=2, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv6')
                 .convact(lfilter_shape=(3,3,128,128), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv7')
                 .deconvact(lfilter_shape=(4,4,64,128), istride=2, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv8')
                 .convact(lfilter_shape=(3,3,64,64), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv9')
                 .sp_attn(self.reservoir['fus_out'], lfilter_shape=(1,1,64,64), spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sname='tr_attn')
                 .convact(lfilter_shape=(3,3,64,32), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='tr_conv10'))
                                                     
            return self.terminals[0]
        
            
    def _Decoder(self, dec_in, name=None, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse=reuse):
            
            (self.feed(dec_in)
                 .convact(lfilter_shape=(3,3,32,32), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='dec_conv1')
                 .convact(lfilter_shape=(3,3,32,16*self.num_classes), istride=1, spadding='REFLECT', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='dec_conv2')
                 .depth_to_space(iblk_size=4, sname='dec_d2s1'))
                                                                                
        return self.terminals[0]
    
    
    def _build(self):
        
        enc_out = self._Encoder(self.image, name='Encoder')
        tr_out = self._Transformer(enc_out, name='Transformer')
        pred = self._Decoder(tr_out, name='Decoder')
        self.reservoir['prediction'] = pred[:, self.start_pos[0]:self.start_pos[0]+self.img_size[0], self.start_pos[1]:self.start_pos[1]+self.img_size[1], :]
        
        self.losses = self._createLoss(name='loss')
        
        #self.IoUs = self._inference()
    def _get_pred(self, inputs):
        
        size = tf.shape(self.image)[1:3]
        pred = tf.image.resize_bilinear(inputs, size=size)
        return tf.argmax(pred, axis=3), tf.cast(tf.multiply(tf.nn.softmax(pred), 255.), dtype=tf.uint8, name='pred_out')
        #return tf.argmax(pred, axis=3)
    

    def _get_mask(self, gt, num_classes, ignore_label):
        
        class_mask = tf.less_equal(gt, num_classes-1)
        not_ignore_mask = tf.not_equal(gt, ignore_label)
        mask = tf.logical_and(class_mask, not_ignore_mask)
        indices = tf.squeeze(tf.where(mask), 1) #Nx2 tensor for indexing right positions
    
        return indices    
 
    
    def _createLoss(self, name, reuse=tf.AUTO_REUSE):
        #####################################
        # ignor label process
        #####################################
        
        gt = tf.reshape(self.label, (-1,))
        indices = self._get_mask(gt, self.num_classes, self.ignore_label) #get label position with not-ignore_label
        gt = tf.cast(tf.gather(gt, indices), tf.int32) # only not-ignore_label ground-truth
        
        pred = tf.reshape(self.reservoir['prediction'], (-1, self.num_classes))
        pred = tf.gather(pred, indices)
                
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt))
                           
        return (loss, )
    
    
    def _confusion_matrix(self, pred, gt): # gt(row)-pred(col) matrix whose element is the count of pixels in that gt-pred
        
        merged_maps = tw.bitwise_or(tw.left_shift(gt, 8), pred)
        hist = tf.bincount(tf.reshape(merged_maps, (-1,)))
        nonzero = tf.squeeze(tf.cast(tf.where(tf.not_equal(hist, 0)), dtype=tf.int32))
        
        pred, gt = tw.bitwise_and(nonzero, 255), tw.right_shift(nonzero, 8)
        
        #class_cnt = tf.maximum(tf.reduce_max(pred), tf.reduce_max(gt)) + 1
        class_cnt = self.num_classes
        indices = class_cnt * gt + pred
        shape = class_cnt * class_cnt
        
        conf_matrix = tf.sparse_to_dense(indices, (shape,), tf.gather(hist, nonzero), 0)
        
        return tf.cast(tf.reshape(conf_matrix, (class_cnt, class_cnt)), dtype=tf.float32)
    
    
    def _mIoU(self, pred, gt):
        
        conf_mat = self._confusion_matrix(pred, gt) #11-person, 12-rider
        
        row_sum = tf.squeeze(tf.reduce_sum(conf_mat, axis=1))
        col_sum = tf.squeeze(tf.reduce_sum(conf_mat, axis=0))
        gt_class_num = tf.cast(tf.count_nonzero(row_sum), dtype=tf.float32)
        diag = tf.squeeze(tf.diag_part(conf_mat))
        
        union = row_sum + col_sum - diag + self.eps
        mIoU = tf.truediv(tf.reduce_sum(tf.truediv(diag, union)), gt_class_num)
        
        return mIoU, conf_mat
    
    
    def _inference(self):
        
        pred = self.reservoir['digits_out']
        
        pred = tf.reshape(pred, (-1,))
        labels = tf.reshape(self.label, (-1,)) #flattening
        
        mask = tf.not_equal(labels, self.ignore_label)
        indices = tf.squeeze(tf.where(mask))
        
        gt = tf.cast(tf.gather(labels, indices), tf.int32)
        pred = tf.cast(tf.gather(pred, indices), tf.int32)
        
        mIoU, conf_mat = self._mIoU(pred, gt)
        
        #person-11, rider-12
        union = tf.reduce_sum(conf_mat[11,:])
        personIoU = tf.cond(tf.equal(union, 0), lambda: 0.0, lambda: tf.truediv(conf_mat[11,11], (union+tf.reduce_sum(conf_mat[:, 11])-conf_mat[11,11]+self.eps)))
        union = tf.reduce_sum(conf_mat[12,:])
        riderIoU = tf.cond(tf.equal(union, 0), lambda: 0.0, lambda: tf.truediv(conf_mat[12,12], (union+tf.reduce_sum(conf_mat[:, 12])-conf_mat[12,12]+self.eps)))
        
        return (mIoU, personIoU, riderIoU)

    
    def optimizer(self):
        
        # linear decay            
        learning_rate = self.lr * (1. - (self.g_step - tf.minimum(self.g_step, self.decay_epoch))/self.lr_width)
        opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = opt.minimize(self.losses[-1])
        
        
        #####################################
        # create session and get handles for iterator selection
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list=self.cfg.visible_device)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        #####################################
        # check-point processing
        self.saver = tf.train.Saver()
        ckpt_loc = self.cfg.ckpt_dir
        self.ckpt_name = os.path.join(ckpt_loc, 'hdtNET_coco')
        
        ckpt = tf.train.get_checkpoint_state(ckpt_loc)
        if ckpt and ckpt.model_checkpoint_path:
            import re
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_step = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("---------------------------------------------------------")
            print(" Success to load checkpoint - {}".format(ckpt_name))
            print(" Session starts at step - {}".format(self.start_step))
            print("---------------------------------------------------------")
        else:
            if not os.path.exists(ckpt_loc):
                os.makedirs(ckpt_loc)
            self.start_step = 0
            print("**********************************************************")
            print("  [*] Failed to find a checkpoint - Start from the first")
            print(" Session starts at step - {}".format(self.start_step))
            print("**********************************************************")
        
        #####################################
        # Summary and Summary Writer
        _ = tf.summary.scalar("Total_Loss", self.sum_loss[0])
        _ = tf.summary.scalar("Evaluation_mIoU", self.sum_loss[1])
        #_ = tf.summary.scalar("Loss-1/4 Loss", self.sum_loss[1])
        #_ = tf.summary.scalar("Loss-1 Loss", self.sum_loss[2])
        #_ = tf.summary.scalar("Mean IoU", self.sum_acc[0])
        #_ = tf.summary.scalar("Person IoU", self.sum_acc[1])
        #_ = tf.summary.scalar("Rider IoU", self.sum_acc[2])
        
        self.summaries = tf.summary.merge_all()
        #self.summaries = tf.summary.scalar("Total Loss", self.sum_loss[0])
        self.writer = tf.summary.FileWriter(self.cfg.log_dir, self.sess.graph)
        
        digits_out = tf.cast(tf.argmax(self.reservoir['prediction'], axis=3), tf.uint8, name='pred_out')
        mIoU, conf_mat = self._mIoU(tf.cast(digits_out, tf.int32), tf.squeeze(self.label, axis=3))

        return train_op, self.losses, self.summaries, digits_out, mIoU


    def save(self, global_step):
                
        self.saver.save(self.sess, self.ckpt_name, global_step)
        print('The checkpoint has been created, step: {}'.format(global_step))
        
    
    
         
        
        
        
        
        
        
        
        
