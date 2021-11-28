# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:58:28 2019

@author: Angelo
"""

import os
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow.compat.v1.bitwise as tw

from network import Network


        
class ICNet(Network):
    
    def __init__(self, cfg, train_reader, eval_reader):
        
        self.cfg = cfg
        self.mode = cfg.mode
        self.num_classes = cfg.param['num_classes']
        self.ignore_label = cfg.param['ignore_label']
        self.loss_weight = (cfg.LAMBDA1, cfg.LAMBDA2, cfg.LAMBDA3)
        self.reservoir = {}
        self.losses = None  # = (loss_sub4, loss_sub2, loss_sub1, total_loss)
        self.start_epoch = 0
        
        self.sum_loss = tf.placeholder(dtype=tf.float32, shape=(5,))
        self.sum_acc = tf.placeholder(dtype=tf.float32, shape=(3,))
        
        self.eps = tf.constant(1e-5)
        
        self.train_reader = train_reader.dataset
        self.eval_reader = eval_reader.dataset
        
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_reader.output_types)
        
        self.images, self.labels = self.iterator.get_next()
        self.images.set_shape([None, None, None, 3])
        self.labels.set_shape([None, None, None, 1])
        
        size = tf.div(tf.add(tf.shape(self.images)[1:3], tf.constant(1)), tf.constant(2))  
        self.images2 = tf.image.resize_bilinear(self.images, size=size, align_corners=True)
        
        self.lr = cfg.LEARNING_RATE
        self.g_step = tf.placeholder(dtype=tf.float32, shape=())
        self.lr_width = float(cfg.TRAIN_EPOCHS - cfg.DECAY_EPOCH)
        self.decay_epoch = float(cfg.DECAY_EPOCH)
    
        super(ICNet, self).__init__()
        
    
    def _ResBottNeck(self, inputs, in_ch, out_ch, strides=1, name=None, reuse=tf.AUTO_REUSE):
        
        ch = tf.to_int32(tf.shape(inputs)[-1])
        
        #shortcut path
        scope = name+'_sc'
        with tf.variable_scope(scope, reuse=reuse):
            
            (self.feed(inputs)
                     .conv(filters=out_ch, kernel_size=1, strides=strides, name='ch_modifier', activation=None)
                     .batch_normalization(name='bn', activation=None))
            
            sc_out = tf.cond(tf.not_equal(ch, out_ch), lambda: self.terminals[0] + 0., lambda: inputs + 0.)
        
        #main_branch path
        scope = name+'_mb'
        with tf.variable_scope(scope, reuse=reuse):
            (self.feed(inputs)
                .conv(filters=in_ch, kernel_size=1, strides=strides, activation=None, name='1x1_1')
                .batch_normalization(name='1x1_1bn')
                .conv(filters=in_ch, kernel_size=3, activation=None, name='3x3_2')
                .batch_normalization(name='3x3_2bn')
                .conv(filters=out_ch, kernel_size=1, name='1x1_3', activation=None)
                .batch_normalization(name='1x1_3bn', activation=None))
            
            mb_out = self.terminals[0] + 0.
        
        with tf.variable_scope(name, reuse=reuse):
            output = sc_out + mb_out
            (self.feed(output)
                 .activator())
            
            return self.terminals[0] + 0.
        
        
    def _ICNetBranch2(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse):
            (self.feed(inputs)
                 .conv(filters=32, kernel_size=3, strides=2, activation=None, name='conv1_1')
                 .batch_normalization(name='conv1_1bn') #inputs, name=None, training=True, activation=DEFAULT_ACTIVATOR, reuse=False
                 .conv(filters=32, kernel_size=3, strides=1, activation=None, name='conv1_2')
                 .batch_normalization(name='conv1_22bn') 
                 .conv(filters=64, kernel_size=3, strides=1, activation=None, name='conv1_3')
                 .batch_normalization(name='conv1_3bn')
                 .max_pool(pool_size=3, name='max_pool'))
            
            x = self._ResBottNeck(self.terminals[0], 32, 128, name='conv2_1')
            x = self._ResBottNeck(x, 32, 128, name='conv2_2')
            x = self._ResBottNeck(x, 32, 128, name='conv2_3')
            
            x = self._ResBottNeck(x, 64, 256, strides=2, name='conv3_1')
            
            self.reservoir['conv3_1'] = x + 0.
            
            return 0

        
    def _PyramidPool(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        size = tf.shape(inputs)[1:3]
        
        pool1 = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        pool1 = tf.image.resize_bilinear(pool1, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(2))
        h, w = h_w[0], h_w[1]
        for ht in range(2):
            for wd in range(2):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool2 = tf.concat(part[:2], axis=2)
        for index in range(1,2):
            pos = index * 2
            row = tf.concat(part[pos:pos+2], axis=2)
            pool2 = tf.concat([pool2, row], axis=1)
        pool2 = tf.image.resize_bilinear(pool2, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(3))
        h, w = h_w[0], h_w[1]
        for ht in range(3):
            for wd in range(3):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool3 = tf.concat(part[:3], axis=2)
        for index in range(1,3):
            pos = index * 3
            row = tf.concat(part[pos:pos+3], axis=2)
            pool3 = tf.concat([pool3, row], axis=1)
        pool3 = tf.image.resize_bilinear(pool3, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(6))
        h, w = h_w[0], h_w[1]
        for ht in range(6):
            for wd in range(6):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool6 = tf.concat(part[:6], axis=2)
        for index in range(1,6):
            pos = index * 6
            row = tf.concat(part[pos:pos+6], axis=2)
            pool6 = tf.concat([pool6, row], axis=1)
        pool6 = tf.image.resize_bilinear(pool6, size=size, align_corners=True)
        
        
        with tf.variable_scope(name, reuse=reuse):
            #sum features
            out = tf.add_n([inputs, pool6, pool3, pool2, pool1])
            
            (self.feed(out)
                 .conv(filters=256, kernel_size=1, strides=1, activation=None, name='1x1Pool')
                 .batch_normalization(name='1x1Poolbn'))
            
            return self.terminals[0] + 0.
        
        
    def _ResBottNeckD(self, inputs, in_ch, out_ch, rate=2, strides=1, name=None, reuse=tf.AUTO_REUSE):
        
        ch = tf.to_int32(tf.shape(inputs)[-1])
        
        #shortcut path
        scope = name+'_sc'
        with tf.variable_scope(scope, reuse=reuse):
            
            (self.feed(inputs)
                     .conv(filters=out_ch, kernel_size=1, strides=strides, name='ch_modifier', activation=None)
                     .batch_normalization(name='bn', activation=None))
            
            sc_out = tf.cond(tf.not_equal(ch, out_ch), lambda: self.terminals[0] + 0., lambda: inputs + 0.)
        
        #main_branch path
        scope = name+'_mb'
        with tf.variable_scope(scope, reuse=reuse):
            (self.feed(inputs)
                .conv(filters=in_ch, kernel_size=1, strides=strides, activation=None, name='1x1_1')
                .batch_normalization(name='1x1_1bn')
                .conv(filters=in_ch, kernel_size=3, rate=rate, activation=None, name='d3x3_d')
                .batch_normalization(name='3x3_2bn')
                .conv(filters=out_ch, kernel_size=1, name='1x1_3', activation=None)
                .batch_normalization(name='1x1_3bn', activation=None))
            
            mb_out = self.terminals[0] + 0.
        
        with tf.variable_scope(name, reuse=reuse):
            output = sc_out + mb_out
            (self.feed(output)
                 .activator())
            
            return self.terminals[0] + 0.

        
    def _ICNetBranch4(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        size = tf.div(tf.add(tf.shape(inputs)[1:3], tf.constant(1)), tf.constant(2))  
                
        with tf.variable_scope(name, reuse=reuse):
            (self.feed(inputs)
                 .resize_bilinear(size=size, name='conv3_1_reduce'))
            
            x = self._ResBottNeck(self.terminals[0], 64, 256, name='conv3_2')
            x = self._ResBottNeck(x, 64, 256, name='conv3_3')
            x = self._ResBottNeck(x, 64, 256, name='conv3_4')
            
            x = self._ResBottNeckD(x, 128, 512, name='conv4_1')
            x = self._ResBottNeckD(x, 128, 512, name='conv4_2')
            x = self._ResBottNeckD(x, 128, 512, name='conv4_3')
            x = self._ResBottNeckD(x, 128, 512, name='conv4_4')
            x = self._ResBottNeckD(x, 128, 512, name='conv4_5')
            x = self._ResBottNeckD(x, 128, 512, name='conv4_6')
            
            x = self._ResBottNeckD(x, 256, 1024, rate=4, name='conv5_1')
            x = self._ResBottNeckD(x, 256, 1024, rate=4, name='conv5_2')
            x = self._ResBottNeckD(x, 256, 1024, rate=4, name='conv5_3')
            
            x = self._PyramidPool(x, name='PyPool')
            
            self.reservoir['conv5_3'] = x + 0.
            
            return 0
        
        
    def _ICNetBranch1(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse=reuse):
            (self.feed(inputs)
                 .conv(filters=32, kernel_size=3, strides=2, activation=None, name='3x3_1')
                 .batch_normalization(name='3x3_1bn') #inputs, name=None, training=True, activation=DEFAULT_ACTIVATOR, reuse=False
                 .conv(filters=32, kernel_size=3, strides=2, activation=None, name='3x3_2')
                 .batch_normalization(name='3x3_2bn') 
                 .conv(filters=64, kernel_size=3, strides=2, activation=None, name='3x3_3')
                 .batch_normalization(name='3x3_3bn') 
                 )
                        
            self.reservoir['conv3'] = self.terminals[0] + 0.
            
            return 0

    
    def _FusionModule(self, small_tensor, large_tensor, s_ch, l_ch, name, reuse=tf.AUTO_REUSE):
        
        large_size = tf.shape(large_tensor)[1:3]
        
        with tf.variable_scope(name, reuse=reuse):
            
            (self.feed(small_tensor)
                 .resize_bilinear(large_size, name='interp'))
            
            self.reservoir[name+'_out'] = self.terminals[0] + 0.
            
            (self.conv_nn(filters=[3,3,s_ch, 128], rate=2, activation=None, name='3x3')
                 .batch_normalization(activation=None, name='3x3bn'))
            
            f_small = self.terminals[0] + 0.
            
            (self.feed(large_tensor)
                 .conv_nn(filters=[1,1,l_ch, 128], activation=None, name='1x1')
                 .batch_normalization(activation=None, name='1x1bn'))
            
            fused = tf.add(f_small, self.terminals[0])
            
            (self.feed(fused)
                 .activator(name='activation'))
            
            return self.terminals[0] + 0.

    
    def _ICNetTail(self, inputs, name, reuse=tf.AUTO_REUSE):
        '''
        h, w = inputs.get_shape().as_list()[1:3]
        size = (2*h, 2*w)
        '''
        size = tf.multiply(tf.shape(inputs)[1:3], tf.constant(2))
        
        with tf.variable_scope(name, reuse=reuse):
            (self.feed(inputs)
                 .resize_bilinear(size, name='interp'))
            
            self.reservoir[name+'_out'] = self.terminals[0] + 0.
        
    
    def _get_mask(self, gt, num_classes, ignore_label):
        
        class_mask = tf.less_equal(gt, num_classes-1)
        not_ignore_mask = tf.not_equal(gt, ignore_label)
        mask = tf.logical_and(class_mask, not_ignore_mask)
        indices = tf.squeeze(tf.where(mask), 1) #Nx2 tensor for indexing right positions
    
        return indices
    
    
    def _get_pred(self, inputs):
        
        size = tf.shape(self.images)[1:3]
        pred = tf.image.resize_bilinear(inputs, size=size)
        return tf.argmax(pred, axis=3)


    def _createLoss(self, name, reuse=tf.AUTO_REUSE):
        #####################################
        #predictions
        #####################################
        tensors = (self.reservoir['sub4_out'], self.reservoir['sub2_out'], self.reservoir['sub1_out'])
        
        losses=[]
        predictions=[]
        labels=[]
        channels=[256, 128, 128]
        
        with tf.variable_scope(name, reuse=reuse):
            for index in range(len(tensors)):
                (self.feed(tensors[index])
                     .conv_nn(filters=[1,1,channels[index],self.num_classes], use_bias=True, activation=None, name='cls_{}'.format(index)))
                
                predictions.append(self.terminals[0] + 0.)
            
            # dummy execution with name = 'pred_out' for graph node generation (freeze out this node)
            self.reservoir['digits_out'] = tf.add(self._get_pred(predictions[-1]), 0, name='pred_out')
            
            #####################################
            #resizing labels 
            #####################################
            for index in range(len(predictions)):
                size =  tf.shape(predictions[index])[1:3]
                (self.feed(self.labels)
                 .resize_nn(size, name='interp_{}'.format(index)))
                labels.append(tf.squeeze(self.terminals[0], axis=[3]))
            
            #####################################
            #ignore-label process and loss calculations
            #####################################
            t_loss = 0.
            for index in range(len(labels)):
                gt = tf.reshape(labels[index], (-1,))
                indices = self._get_mask(gt, self.num_classes, self.ignore_label) #get label position with not-ignore_label
                gt = tf.cast(tf.gather(gt, indices), tf.int32) # only not-ignore_label ground-truth
                
                pred = tf.reshape(predictions[index], (-1, self.num_classes))
                pred = tf.gather(pred, indices) # only not-ignore_label prediction
                
                loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt))
                t_loss += self.loss_weight[index] * loss
                losses.append(loss)
                
            losses.append(t_loss)
            
            return losses
        
    
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
        labels = tf.reshape(self.labels, (-1,)) #flattening
        
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
        #####################################
        # weight-decay, learning-rate control, optimizer selection with bn training
        #####################################
        if self.cfg.WEIGHT_DECAY != 0.0:
            l2_weight = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if ('bn' not in var.name)])
            loss_to_opt = self.losses[-1] + self.cfg.WEIGHT_DECAY * l2_weight
            self.losses.append(loss_to_opt)
        
        # linear decay            
        learning_rate = self.lr * (1. - (self.g_step - tf.minimum(self.g_step, self.decay_epoch))/self.lr_width)
        opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
        if self.cfg.BN_LEARN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        else:
            update_ops = None
        train_op = opt.minimize(self.losses[-1])
        train_op = tf.group([train_op, update_ops])
        
        #####################################
        # create session and get handles for iterator selection
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list='0')
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        self.train_handle = self.sess.run(self.train_reader.string_handle())
        self.eval_handle = self.sess.run(self.eval_reader.string_handle())
        #####################################
        # check-point processing
        self.saver = tf.train.Saver()
        ckpt_loc = self.cfg.ckpt_dir
        self.ckpt_name = os.path.join(ckpt_loc, 'ICnetModel')
        
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
        _ = tf.summary.scalar("Total_Loss", self.sum_loss[3])
        _ = tf.summary.scalar("Branch-4 Loss", self.sum_loss[0])
        _ = tf.summary.scalar("Branch-2 Loss", self.sum_loss[1])
        _ = tf.summary.scalar("Branch-1 Loss", self.sum_loss[2])
        _ = tf.summary.scalar("Mean IoU", self.sum_acc[0])
        _ = tf.summary.scalar("Person IoU", self.sum_acc[1])
        _ = tf.summary.scalar("Rider IoU", self.sum_acc[2])
        
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.cfg.log_dir, self.sess.graph)
        
        #####################################
        # Inference and Evaluation Measure
        IoUs = self._inference()
        Images = (self.images, self.labels)

        return train_op, self.losses, self.summaries, self.reservoir['digits_out'], IoUs, Images


    def save(self, global_step):
                
        self.saver.save(self.sess, self.ckpt_name, global_step)
        print('The checkpoint has been created, step: {}'.format(global_step))
    
        
        
    def _build(self):
        
        self._ICNetBranch2(self.images2, name='br2')
        self._ICNetBranch4(self.reservoir['conv3_1'], name='br4')
        self._ICNetBranch1(self.images, name='br1')
        
        x = self._FusionModule(self.reservoir['conv5_3'], self.reservoir['conv3_1'], s_ch=256, l_ch=256, name='sub4') #reservior holds sub4_out tensor for loss cal.
        x = self._FusionModule(x, self.reservoir['conv3'], s_ch=128, l_ch=64, name='sub2') #reservoir hods sub2_out tensor for loss cal.
        
        self._ICNetTail(x, name='sub1') #reservoir holds sub1_out tensor for loss cal.
        
        self.losses = self._createLoss(name='loss')
        
        
class hdtNET_MN2(Network):
    
    def __init__(self, cfg, train_reader, eval_reader):
        
        self.cfg = cfg
        self.num_classes = cfg.param['num_classes']
        self.ignore_label = cfg.param['ignore_label']
        self.loss_weight = (cfg.LAMBDA16, cfg.LAMBDA4, cfg.LAMBDA1)
        self.reservoir = {}
        self.losses = None  # = (loss_sub4, loss_sub2, loss_sub1, total_loss)
        
        self.sum_loss = tf.placeholder(dtype=tf.float32, shape=(4,))
        self.sum_acc = tf.placeholder(dtype=tf.float32, shape=(3,))
        
        self.eps = tf.constant(1e-5)
        
        self.train_reader = train_reader.dataset
        self.eval_reader = eval_reader.dataset
        
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_reader.output_types)
        
        self.image, self.label = self.iterator.get_next()
        self.image.set_shape([None, None, None, 3])
        self.label.set_shape([None, None, None, 1])
        
        self.lr = cfg.LEARNING_RATE
        self.g_step = tf.placeholder(dtype=tf.float32, shape=())
        self.lr_width = float(cfg.TRAIN_EPOCHS - cfg.DECAY_EPOCH)
        self.decay_epoch = float(cfg.DECAY_EPOCH)
    
        super(hdtNET_MN2, self).__init__()
        
    
    def _MN2btneck(self, inputs, Cin, Cout, t=6, st=1, r=1, name=None, reuse=tf.AUTO_REUSE):
        
        Ctmp = t*Cin
        Side = tf.cond(tf.equal(st, 1), lambda: inputs + 0., lambda: 0.)
        Side = tf.cond(tf.equal(Cin, Cout), lambda: Side, lambda: 0.)
        
        with tf.variable_scope(name, reuse):
            (self.feed(inputs)
                 .conv_nn(filters=(1,1,Cin, Ctmp), activation=None, name='conv1x1_1')
                 .batch_normalization(name='conv1x1_1bn', activation=tf.nn.relu6)
                 .conv_dw(filters=(3,3,Ctmp,1), strides=(1,st,st,1), rate=(r,r), activation=None, name='conv3dw')
                 .batch_normalization(name='conv3dw_bn', activation=tf.nn.relu6)
                 .conv_nn(filters=(1,1,Ctmp,Cout), activation=None, name='conv1x1_2')
                 .batch_normalization(name='conv1x1_2bn', activation=None))
            
        return self.terminals[0] + Side
            
            
    #inputs, filters, rate=1, strides=[1,1,1,1], padding='SAME', activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER, name=None)
    def _HeadBranch(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse):
            (self.feed(inputs)
                 .conv_nn(filters=(3, 3, 3, 32), strides=(1, 2, 2, 1), activation=None, name='convh_1')
                 .batch_normalization(name='convh_1bn')) #inputs, name=None, training=True, activation=DEFAULT_ACTIVATOR, reuse=False
                 
            x = self._MN2btneck(self.terminals[0], 32, 16, t=1, st=1, r=1, name='btn1')
            x = self._MN2btneck(x, 16, 24, t=6, st=2, r=1, name='btn2_1')
            x = self._MN2btneck(x, 24, 24, t=6, st=1, r=1, name='btn2_2')
            
            self.reservoir['head_out'] = x + 0.
            
        return 0
        
    def _BodyBranch(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        x = inputs + 0.
        
        with tf.variable_scope(name, reuse):
            
            x = self._MN2btneck(x, 24, 32, t=6, st=2, r=1, name='btn1_1')
            x = self._MN2btneck(x, 32, 32, t=6, st=1, r=1, name='btn1_2')
            x = self._MN2btneck(x, 32, 32, t=6, st=1, r=1, name='btn1_3')
            
            x = self._MN2btneck(x, 32, 48, t=6, st=2, r=1, name='btn2_1')
            x = self._MN2btneck(x, 48, 48, t=6, st=1, r=1, name='btn2_2')
            x = self._MN2btneck(x, 48, 48, t=6, st=1, r=1, name='btn2_3')
            x = self._MN2btneck(x, 48, 48, t=6, st=1, r=1, name='btn2_4')
            
            x = self._MN2btneck(x, 48, 64, t=6, st=1, r=1, name='btn3_1')
            x = self._MN2btneck(x, 64, 64, t=6, st=1, r=2, name='btn3_2')
            x = self._MN2btneck(x, 64, 64, t=6, st=1, r=4, name='btn3_3')
            
            x = self._MN2btneck(x, 64, 96, t=6, st=1, r=1, name='btn4_1')
            x = self._MN2btneck(x, 96, 96, t=6, st=1, r=1, name='btn4_2')
            x = self._MN2btneck(x, 96, 96, t=6, st=1, r=1, name='btn4_3')
            
            x = self._MN2btneck(x, 96, 128, t=6, st=1, r=1, name='btn5_1')
            x = self._MN2btneck(x, 128, 128, t=6, st=1, r=1, name='btn5_2')
            x = self._MN2btneck(x, 128, 128, t=6, st=1, r=1, name='btn5_3')
            
            x = self._MN2btneck(x, 128, 160, t=6, st=1, r=1, name='btn6_1')
            x = self._MN2btneck(x, 160, 160, t=6, st=1, r=2, name='btn6_2')
            x = self._MN2btneck(x, 160, 160, t=6, st=1, r=4, name='btn6_3')
            
            x = self._MN2btneck(x, 160, 240, t=6, st=1, r=1, name='btn7_1')
            x = self._MN2btneck(x, 240, 240, t=6, st=1, r=1, name='btn7_2')
            x = self._MN2btneck(x, 240, 240, t=6, st=1, r=1, name='btn7_3')
            
            x = self._MN2btneck(x, 240, 320, t=6, st=1, r=1, name='btn8_1')
            x = self._MN2btneck(x, 320, 320, t=6, st=1, r=1, name='btn8_2')
            x = self._MN2btneck(x, 320, 320, t=6, st=1, r=1, name='btn8_3')
            
        self.reservoir['body_out'] = x + 0.
        
        return 0
        
    
    def _TailBranch(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        size = tf.shape(inputs)[1:3]
        
        pool1 = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        pool1 = tf.image.resize_bilinear(pool1, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(2))
        h, w = h_w[0], h_w[1]
        for ht in range(2):
            for wd in range(2):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool2 = tf.concat(part[:2], axis=2)
        for index in range(1,2):
            pos = index * 2
            row = tf.concat(part[pos:pos+2], axis=2)
            pool2 = tf.concat([pool2, row], axis=1)
        pool2 = tf.image.resize_bilinear(pool2, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(3))
        h, w = h_w[0], h_w[1]
        for ht in range(3):
            for wd in range(3):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool3 = tf.concat(part[:3], axis=2)
        for index in range(1,3):
            pos = index * 3
            row = tf.concat(part[pos:pos+3], axis=2)
            pool3 = tf.concat([pool3, row], axis=1)
        pool3 = tf.image.resize_bilinear(pool3, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(6))
        h, w = h_w[0], h_w[1]
        for ht in range(6):
            for wd in range(6):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool6 = tf.concat(part[:6], axis=2)
        for index in range(1,6):
            pos = index * 6
            row = tf.concat(part[pos:pos+6], axis=2)
            pool6 = tf.concat([pool6, row], axis=1)
        pool6 = tf.image.resize_bilinear(pool6, size=size, align_corners=True)
        
        
        with tf.variable_scope(name, reuse=reuse):
            #sum features
            out = tf.add_n([inputs, pool6, pool3, pool2, pool1])
            out = self._MN2btneck(out, 320, 160, t=6, st=1, r=1, name='btn1')
            
            self.reservoir['tail_out'] = out
            
        return 0    
    
    
    def _FusionModule(self, small_tensor, large_tensor, s_ch, l_ch, name, reuse=tf.AUTO_REUSE):
        
        large_size = tf.shape(large_tensor)[1:3]
        
        with tf.variable_scope(name, reuse=reuse):
            
            (self.feed(small_tensor)
                 .resize_bilinear(large_size, name='interp'))
            
            small_prj = self._MN2btneck(self.terminals[0], s_ch, 48, t=6, st=1, r=1, name='int-prj')
            large_prj = self._MN2btneck(large_tensor, l_ch, 48, t=6, st=1, r=1, name='lrg_prj')
            
            fused = tf.add(small_prj, large_prj)
            
            (self.feed(fused)
                 .activator(name='activation'))
            
        return self.terminals[0] + 0.
    
    
    def _get_pred(self, inputs):
        
        size = tf.shape(self.image)[1:3]
        pred = tf.image.resize_bilinear(inputs, size=size)
        return tf.argmax(pred, axis=3)
    

    def _get_mask(self, gt, num_classes, ignore_label):
        
        class_mask = tf.less_equal(gt, num_classes-1)
        not_ignore_mask = tf.not_equal(gt, ignore_label)
        mask = tf.logical_and(class_mask, not_ignore_mask)
        indices = tf.squeeze(tf.where(mask), 1) #Nx2 tensor for indexing right positions
    
        return indices    
 
    
    def _createLoss(self, name, reuse=tf.AUTO_REUSE):
        #####################################
        #predictions
        #####################################
        tail_out = self.reservoir['tail_out'] + 0.
        head_out = self.reservoir['head_out'] + 0.
        
        losses=[]
        predictions=[]
        labels=[]
                
        fused = self._FusionModule(tail_out, head_out, 160, 24, name='fusion')
        
        with tf.variable_scope(name, reuse=reuse):
            
            tail_out = self._MN2btneck(tail_out, 160, 48, t=6, st=1, r=1, name='tail_prj')
            
            (self.feed(tail_out)
                 .conv_nn(filters=(3,3,48,self.num_classes), use_bias=True, activation=None, name='cls_tail'))
            predictions.append(self.terminals[0] + 0.)
            
            (self.feed(fused)
                 .conv_nn(filters=(3,3,48,self.num_classes), use_bias=True, activation=None, name='cls_head'))
            predictions.append(self.terminals[0] + 0.)
            
            # dummy execution with name = 'pred_out' for graph node generation (freeze out this node)
            #self.reservoir['digits_out'] = tf.add(tf.argmax(self.terminals[0], axis=3), 0, name='pred_out')
            self.reservoir['digits_out'] = tf.add(self._get_pred(predictions[-1]), 0, name='pred_out')
            
            #####################################
            #resizing labels 
            #####################################
            for index in range(len(predictions)):
                size =  tf.shape(predictions[index])[1:3]
                (self.feed(self.label)
                     .resize_nn(size, name='interp_{}'.format(index)))
                labels.append(tf.squeeze(self.terminals[0], axis=[3]))
            
            #####################################
            #ignore-label process and loss calculations
            #####################################
            t_loss = 0.
            for index in range(len(labels)):
                gt = tf.reshape(labels[index], (-1,))
                indices = self._get_mask(gt, self.num_classes, self.ignore_label) #get label position with not-ignore_label
                gt = tf.cast(tf.gather(gt, indices), tf.int32) # only not-ignore_label ground-truth
                
                pred = tf.reshape(predictions[index], (-1, self.num_classes))
                pred = tf.gather(pred, indices) # only not-ignore_label prediction
                
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt))
                t_loss += self.loss_weight[index] * loss
                losses.append(loss)
                
            losses.append(t_loss)
            
            return losses
    
    
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
        #####################################
        # weight-decay, learning-rate control, optimizer selection with bn training
        #####################################
        if self.cfg.WEIGHT_DECAY != 0.0:
            l2_weight = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if (('bn' not in var.name) and ('conv3dw' not in var.name))])
            loss_to_opt = self.losses[-1] + self.cfg.WEIGHT_DECAY * l2_weight
            self.losses.append(loss_to_opt)
        
        # linear decay            
        learning_rate = self.lr * (1. - (self.g_step - tf.minimum(self.g_step, self.decay_epoch))/self.lr_width)
        opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
        if self.cfg.BN_LEARN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        else:
            update_ops = None
        train_op = opt.minimize(self.losses[-1])
        train_op = tf.group([train_op, update_ops])
        
        #####################################
        # create session and get handles for iterator selection
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list='0')
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        self.train_handle = self.sess.run(self.train_reader.string_handle())
        self.eval_handle = self.sess.run(self.eval_reader.string_handle())
        #####################################
        # check-point processing
        self.saver = tf.train.Saver()
        ckpt_loc = self.cfg.ckpt_dir
        self.ckpt_name = os.path.join(ckpt_loc, 'hdtNET_MN2')
        
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
        _ = tf.summary.scalar("Total_Loss", self.sum_loss[2])
        _ = tf.summary.scalar("Loss-1/16", self.sum_loss[0])
        _ = tf.summary.scalar("Loss-1/4 Loss", self.sum_loss[1])
        #_ = tf.summary.scalar("Loss-1 Loss", self.sum_loss[2])
        _ = tf.summary.scalar("Mean IoU", self.sum_acc[0])
        _ = tf.summary.scalar("Person IoU", self.sum_acc[1])
        _ = tf.summary.scalar("Rider IoU", self.sum_acc[2])
        
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.cfg.log_dir, self.sess.graph)
        

        return train_op, self.losses, self.summaries, self.reservoir['digits_out'], self.IoUs, self.Images


    def save(self, global_step):
                
        self.saver.save(self.sess, self.ckpt_name, global_step)
        print('The checkpoint has been created, step: {}'.format(global_step))
        
    
    def _build(self):
        
        self._HeadBranch(self.image, name='head')
        self._BodyBranch(self.reservoir['head_out'], name='body')
        self._TailBranch(self.reservoir['body_out'], name='tail')
        
        self.losses = self._createLoss(name='loss')
        
        self.IoUs = self._inference()
        self.Images = (self.image, self.label)

        
class hdtNET(Network):
    
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.num_classes = cfg.param['num_classes']
        self.ignore_label = cfg.param['ignore_label']
        self.loss_weight = (cfg.LAMBDA16, cfg.LAMBDA4, cfg.LAMBDA1)
        self.reservoir = {}
        self.losses = None  # = (loss_sub4, loss_sub2, loss_sub1, total_loss)
        
        self.sum_loss = tf.placeholder(dtype=tf.float32, shape=(4,))
        
        self.eps = tf.constant(1e-5)
        
        self.img_in = tf.placeholder(tf.uint8)
        self.gt_in = tf.placeholder(tf.uint8)
        # dummy operation to replace input node for inference after training
        self.image = tf.subtract(tf.cast(self.img_in, tf.float32), cfg.IMG_MEAN)
        self.label = tf.div(tf.cast(self.gt_in, tf.int32), 255)
        
        self.image.set_shape([None, None, None, 3])
        self.label.set_shape([None, None, None, 1])
        
        self.img_size = cfg.TRAIN_SIZE[0]*cfg.TRAIN_SIZE[1]
        self.batch_size = cfg.BATCH_SIZE
        self.person_weight = cfg.person_weight
        
        self.lr = cfg.LEARNING_RATE
        self.g_step = tf.placeholder(dtype=tf.float32, shape=())
        self.lr_width = float(cfg.TRAIN_EPOCHS - cfg.DECAY_EPOCH)
        self.decay_epoch = float(cfg.DECAY_EPOCH)
        
        #self.scale = 100000.
    
        super(hdtNET, self).__init__()
        
    
    def _MN2btneck(self, inputs, Cin, Cout, t=6, st=1, r=1, f=3, name=None, reuse=tf.AUTO_REUSE):
        
        Ctmp = t*Cin
        Side = tf.cond(tf.equal(st, 1), lambda: inputs + 0., lambda: 0.)
        Side = tf.cond(tf.equal(Cin, Cout), lambda: Side, lambda: 0.)
        
        with tf.variable_scope(name, reuse):
            (self.feed(inputs)
                 .conv_nn(filters=(1,1,Cin, Ctmp), activation=None, name='conv1x1_1')
                 .batch_normalization(name='conv1x1_1bn', activation=tf.nn.relu6)
                 .conv_dw(filters=(f,f,Ctmp,1), strides=(1,st,st,1), rate=(r,r), activation=None, padding='REFLECT', name='conv3dw')
                 .batch_normalization(name='conv3dw_bn', activation=tf.nn.relu6)
                 .conv_nn(filters=(1,1,Ctmp,Cout), activation=None, name='conv1x1_2')
                 .batch_normalization(name='conv1x1_2bn', activation=None))
            
        return self.terminals[0] + Side
            
            
    #inputs, filters, rate=1, strides=[1,1,1,1], padding='SAME', activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER, name=None)
    def _HeadBranch(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        with tf.variable_scope(name, reuse):
            (self.feed(inputs)
                 .conv_nn(filters=(3,3,3,16), strides=(1,2,2,1), activation=None, padding='REFLECT', name='conv1_1')
                 .batch_normalization(name='conv1_1bn')
                 #.conv_dw(filters=(5,5,16,1), strides=(1,1,1,1), rate=(1,1), activation=None, name='conv1_2_conv3dw')
                 #.batch_normalization(name='conv1_2bn')
                 #.conv_nn(filters=(3,3,32,32), strides=(1,1,1,1), activation=None, name='conv1_3')
                 #.batch_normalization(name='conv1_3bn')
                 .conv_nn(filters=(3,3,16,64), strides=(1,2,2,1), activation=None, padding='REFLECT', name='conv2_1')
                 .batch_normalization(name='conv2_1bn'))
                 #.conv_dw(filters=(11,11,64,1), strides=(1,1,1,1), rate=(1,1), activation=None, name='conv2_2_conv3dw')
                 #.batch_normalization(name='conv2_2bn')
                 #.conv_nn(filters=(3,3,64,64), strides=(1,1,1,1), activation=None, name='conv2_3')
                 #.batch_normalization(name='conv2_3bn')) #inputs, name=None, training=True, activation=DEFAULT_ACTIVATOR, reuse=False
                 
        self.reservoir['head_out'] = self.terminals[0] + 0.
            
        return 0
        
    def _BodyBranch(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        x = inputs + 0.
        
        with tf.variable_scope(name, reuse):
            
            x = self._MN2btneck(x, 64, 32, t=2, st=2, r=1, name='btn3_1')
            x = self._MN2btneck(x, 32, 32, t=4, st=1, r=1, f=7, name='btn3_2')
            x = self._MN2btneck(x, 32, 32, t=4, st=1, r=1, f=11, name='btn3_3')
            
            x = self._MN2btneck(x, 32, 64, t=8, st=2, r=1, name='btn4_1')
            x = self._MN2btneck(x, 64, 64, t=4, st=1, r=1, f=7, name='btn4_2')
            x = self._MN2btneck(x, 64, 64, t=4, st=1, r=1, f=11, name='btn4_3')
            x = self._MN2btneck(x, 64, 64, t=4, st=1, r=1, f=15, name='btn4_4')
            
            x = self._MN2btneck(x, 64, 128, t=8, st=1, r=1, name='btn5_1')
            x = self._MN2btneck(x, 128, 128, t=4, st=1, r=1, f=7, name='btn5_2')
            x = self._MN2btneck(x, 128, 128, t=4, st=1, r=1, f=11, name='btn5_3')
            x = self._MN2btneck(x, 128, 128, t=4, st=1, r=1, f=15, name='btn5_4')
            
        self.reservoir['body_out'] = x + 0.
        
        return 0
        
    
    def _TailBranch(self, inputs, name, reuse=tf.AUTO_REUSE):
        
        size = tf.shape(inputs)[1:3]
        
        pool1 = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        pool1 = tf.image.resize_bilinear(pool1, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(2))
        h, w = h_w[0], h_w[1]
        for ht in range(2):
            for wd in range(2):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool2 = tf.concat(part[:2], axis=2)
        for index in range(1,2):
            pos = index * 2
            row = tf.concat(part[pos:pos+2], axis=2)
            pool2 = tf.concat([pool2, row], axis=1)
        pool2 = tf.image.resize_bilinear(pool2, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(3))
        h, w = h_w[0], h_w[1]
        for ht in range(3):
            for wd in range(3):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool3 = tf.concat(part[:3], axis=2)
        for index in range(1,3):
            pos = index * 3
            row = tf.concat(part[pos:pos+3], axis=2)
            pool3 = tf.concat([pool3, row], axis=1)
        pool3 = tf.image.resize_bilinear(pool3, size=size, align_corners=True)
        
        part = []
        h_w = tf.div(size, tf.constant(6))
        h, w = h_w[0], h_w[1]
        for ht in range(6):
            for wd in range(6):
                id_h1, id_h2 = h * ht, h*(ht+1)
                id_w1, id_w2 = w * wd, w*(wd+1)
                part.append(tf.reduce_mean(inputs[:, id_h1:id_h2, id_w1:id_w2, :], axis=[1, 2], keepdims=True))
        
        pool6 = tf.concat(part[:6], axis=2)
        for index in range(1,6):
            pos = index * 6
            row = tf.concat(part[pos:pos+6], axis=2)
            pool6 = tf.concat([pool6, row], axis=1)
        pool6 = tf.image.resize_bilinear(pool6, size=size, align_corners=True)
        
        
        with tf.variable_scope(name, reuse=reuse):
            #sum features
            out = tf.add_n([inputs, pool6, pool3, pool2, pool1])
            out = self._MN2btneck(out, 128, 64, t=2, st=1, r=1, name='btn_pool')
            '''
            (self.feed(out)
                 .conv_nn(filters=(3,3,128,64), strides=(1,1,1,1), activation=None, name='btn_pool')
                 .batch_normalization(name='btn_poolbn'))
            '''
            self.reservoir['tail_out'] = out
            
        return 0    
    
    
    def _FusionModule(self, small_tensor, large_tensor, s_ch, l_ch, name, reuse=tf.AUTO_REUSE):
        
        large_size = tf.shape(large_tensor)[1:3]
        
        with tf.variable_scope(name, reuse=reuse):
            
            (self.feed(small_tensor)
                 .resize_bilinear(large_size, name='interp')
                 .conv_nn(filters=(3,3,s_ch,s_ch), strides=(1,1,1,1), activation=None, padding='REFLECT', name='conv_interp') #dw_conv with larger tabs maybe here
                 .batch_normalization(name='conv_interpbn', activation=None))
            
            fused = tf.add(self.terminals[0], large_tensor)
            
            (self.feed(fused)
                 .activator(name='activation'))
            
        return self.terminals[0] + 0.
    
    
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
        #predictions
        #####################################
        tail_out = self.reservoir['tail_out'] + 0.
        head_out = self.reservoir['head_out'] + 0.
        
        losses=[]
        predictions=[]
        labels=[]
                
        (self.feed(head_out)
             .conv_nn(filters=(3,3,64,64), activation=None, padding='REFLECT', name='convF')
             .batch_normalization(name='convFbn'))
        
        #head_out = self.terminals[0] + 0.
        
        fused = self._FusionModule(tail_out, self.terminals[0], 64, 64, name='fusion')
        
        with tf.variable_scope(name, reuse=reuse):
            
            (self.feed(self.reservoir['tail_out']) #self.feed(tail_out)
                 .conv_nn(filters=(1,1,64,self.num_classes), use_bias=True, activation=None, name='cls_tail'))
            predictions.append(self.terminals[0] + 0.)
            
            (self.feed(fused)
                 .conv_nn(filters=(1,1,64,self.num_classes), use_bias=True, activation=None, name='cls_head'))
            predictions.append(self.terminals[0] + 0.)
            
            # dummy execution with name = 'pred_out' for graph node generation (freeze out this node)
            #self.reservoir['digits_out'] = tf.add(tf.argmax(self.terminals[0], axis=3), 0, name='pred_out')
            self.reservoir['digits_out'], _ = self._get_pred(predictions[-1])
            
            #####################################
            #resizing labels 
            #####################################
            for index in range(len(predictions)):
                size =  tf.shape(predictions[index])[1:3]
                (self.feed(self.label)
                     .resize_nn(size, name='interp_{}'.format(index)))
                labels.append(tf.squeeze(self.terminals[0], axis=[3]))
            
            #####################################
            #ignore-label process and loss calculations
            #####################################
            t_loss = 0.
            for index in range(len(labels)):
                gt = tf.reshape(labels[index], (-1,))
                indices = self._get_mask(gt, self.num_classes, self.ignore_label) #get label position with not-ignore_label
                gt = tf.cast(tf.gather(gt, indices), tf.int32) # only not-ignore_label ground-truth
                
                pred = tf.reshape(predictions[index], (-1, self.num_classes))
                pred = tf.gather(pred, indices) # only not-ignore_label prediction
                
                ###########################################
                # WEIGHT PROCESS
                ###########################################
                b_weight = tf.reduce_sum(labels[index], axis=(1,2))
                b_weight = tf.where(tf.equal(b_weight, 0), tf.multiply(tf.ones(tf.shape(b_weight), dtype=tf.int32), self.img_size), b_weight)
                b_weight = tf.subtract(tf.truediv(self.img_size, b_weight), 1.)
                
                class_weights = (1.0, tf.clip_by_value(self.person_weight*b_weight[0], 1.0, 2.0))
                weights = tf.expand_dims(tf.gather(class_weights, labels[index][0]), 0)
                
                for b_index in range(1, self.batch_size):
                    class_weights = (1.0, tf.clip_by_value(self.person_weight*b_weight[b_index], 1.0, 2.0))
                    weights = tf.concat((weights, tf.expand_dims(tf.gather(class_weights, labels[index][b_index]), axis=0)), axis=0)
                
                weights = tf.reshape(weights, (-1,))
                weights = tf.gather(weights, indices)
                loss = tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=gt, weights=weights)
                #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt))
                
                t_loss += self.loss_weight[index] * loss
                losses.append(loss)
                
            losses.append(t_loss)
            
            return losses
    
    
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
        #####################################
        # weight-decay, learning-rate control, optimizer selection with bn training
        #####################################
        if self.cfg.WEIGHT_DECAY != 0.0:
            l2_weight = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if (('bn' not in var.name) and ('conv3dw' not in var.name))])
            loss_to_opt = self.losses[-1] + self.cfg.WEIGHT_DECAY * l2_weight
            self.losses.append(loss_to_opt)
        
        # linear decay            
        learning_rate = self.lr * (1. - (self.g_step - tf.minimum(self.g_step, self.decay_epoch))/self.lr_width)
        opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
        if self.cfg.BN_LEARN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        else:
            update_ops = None
        train_op = opt.minimize(self.losses[-1])
        train_op = tf.group([train_op, update_ops])
        
        #####################################
        # create session and get handles for iterator selection
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list='0')
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
        _ = tf.summary.scalar("Total_Loss", self.sum_loss[2])
        _ = tf.summary.scalar("Loss-1/16", self.sum_loss[0])
        _ = tf.summary.scalar("Loss-1/4 Loss", self.sum_loss[1])
        #_ = tf.summary.scalar("Loss-1 Loss", self.sum_loss[2])
        #_ = tf.summary.scalar("Mean IoU", self.sum_acc[0])
        #_ = tf.summary.scalar("Person IoU", self.sum_acc[1])
        #_ = tf.summary.scalar("Rider IoU", self.sum_acc[2])
        
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.cfg.log_dir, self.sess.graph)
        

        return train_op, self.losses, self.summaries, tf.cast(self.reservoir['digits_out'], tf.uint8)


    def save(self, global_step):
                
        self.saver.save(self.sess, self.ckpt_name, global_step)
        print('The checkpoint has been created, step: {}'.format(global_step))
        
    
    def _build(self):
        
        self._HeadBranch(self.image, name='head')
        self._BodyBranch(self.reservoir['head_out'], name='body')
        self._TailBranch(self.reservoir['body_out'], name='tail')
        
        self.losses = self._createLoss(name='loss')
        
        #self.IoUs = self._inference()
        
        
        
        
        
        