import tensorflow.compat.v1 as tf
from network import Network
import os

tf.disable_eager_execution()

class DenseNet(Network):
    
    def __init__(self, cfg):
        
        self.batch_img = tf.placeholder(dtype=tf.uint8) # CIFAR-10 image batch
        self.batch_lab = tf.placeholder(dtype=tf.uint8) # CIFAR-10 label batch
        self.labels = tf.cast(self.batch_lab, tf.int32)
        self.btrain = tf.placeholder(dtype=tf.bool)
        self.breset = tf.placeholder(dtype=tf.bool)
        self.buse_drop = tf.placeholder(dtype=tf.bool)
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        
        self.reservoir = {} #empty dict for loss related tensors
        
        self.cfg = cfg
                
        super(DenseNet, self).__init__()
        
    def _inference(self, tin):
        
        act = 'SWISH'
        init = 'he_normal'
        
        inputs = tf.cast(tf.reshape(tin, (-1, 3072)), tf.float32)/255. -0.5  # make input as float32 and in the range (-0.5, 0.5)
        #dense(self, tinputs, iin_nodes=10, iout_nodes=10, buse_bias=True, sinitializer='he_uniform', sname=None):
        (self.feed(inputs)
             .denseact(iin_nodes=3072, iout_nodes=768, buse_bias=True, sinitializer=init, sactivation=act, sname='hidden-1')
             .denseact(iin_nodes=768, iout_nodes=512, buse_bias=True, sinitializer=init, sactivation=act, sname='hidden-2')
             .denseact(iin_nodes=512, iout_nodes=256, buse_bias=True, sinitializer=init, sactivation=act, sname='hidden-3')
             .dense(iin_nodes=256, iout_nodes=10, buse_bias=True, sinitializer=init, sname='out')) # num_label output without activation
        
        return self.terminals[0]
        
        
    def _build(self):
        
        ########################################################
        # NEEDED OPERATIONS FOR CALCULATING LOSS
        ########################################################
        self.reservoir['logits'] = self._inference(self.batch_img)
                
        self.losses = self._createLoss(name='loss')
    
    
    def _createLoss(self, name=None, reuse=tf.AUTO_REUSE):
        ###################################################
        # LOSS CALCULATION
        ##################################################
        with tf.variable_scope(name, reuse=reuse):
            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.reservoir['logits'])    
            loss = tf.reduce_mean(xentropy)
                        
        return loss
            
    
    def optimizer(self):
        
        ####################################################
        # OPTIMIZERS (i.e., Adam or RMSProp, etc) SETTING
        #####################################################
        opt = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate)
        train_op = opt.minimize(self.losses)
                        
        ############################################################
        # create session 
        ############################################################
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        # when multiple GPUs - gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
                                
        return train_op, self.losses, self.losses, self.reservoir['logits']
    
    
        
class ConvNet(Network):
    
    def __init__(self, cfg):
        
        self.batch_img = tf.placeholder(dtype=tf.uint8) # CIFAR-10 image batch
        self.batch_lab = tf.placeholder(dtype=tf.uint8) # CIFAR-10 label batch
        self.labels = tf.cast(self.batch_lab, tf.int32)
        self.btrain = tf.placeholder(dtype=tf.bool)
        self.breset = tf.placeholder(dtype=tf.bool)
        self.buse_drop = tf.placeholder(dtype=tf.bool)
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        
        self.reservoir = {} #empty dict for loss related tensors
        
        self.cfg = cfg
                
        super(ConvNet, self).__init__()
        
    def _inference(self, tin):
        
        inputs = tf.cast(self.batch_img, tf.float32)/255. - 0.5 # make input as float32 and in the range (-0.5, 0.5)
        padding='REFLECT'
        act='SWISH' #provides better performance than ReLu
        init='he_normal'
        
        #conv(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sname=None):
        (self.feed(inputs)
             .convact(lfilter_shape=(3,3,3,16), spadding=padding, sinitializer=init, sactivation=act, sname='conv1')
             .convact(lfilter_shape=(3,3,16,16), spadding=padding, sinitializer=init, sactivation=act, sname='conv1-1')
             .maxpool(sname='pool1')
             .convact(lfilter_shape=(3,3,16,64), spadding=padding, sinitializer=init, sactivation=act, sname='conv2')
             .convact(lfilter_shape=(3,3,64,64), spadding=padding, sinitializer=init, sactivation=act, sname='conv2-1')
             .maxpool(sname='pool2')
             .convact(lfilter_shape=(3,3,64,128), spadding=padding, sinitializer=init, sactivation=act, sname='conv3')
             .convact(lfilter_shape=(3,3,128,128), spadding=padding, sinitializer=init, sactivation=act, sname='conv3-1')
             .maxpool(sname='pool3')
             .convact(lfilter_shape=(3,3,128,256), spadding=padding, sinitializer=init, sactivation=act, sname='conv4')
             .convact(lfilter_shape=(3,3,256,256), spadding=padding, sinitializer=init, sactivation=act, sname='conv4-1')
             .maxpool(sname='pool4')
             .convact(lfilter_shape=(2,2,256,256), spadding='VALID', sinitializer=init, sactivation=act, sname='FC1')
             .conv(lfilter_shape=(1,1,256,10), spadding='VALID', buse_bias=True, sinitializer=init, sname='out'))
        
        return self.terminals[0]
        
        
    def _build(self):
        
        ########################################################
        # NEEDED OPERATIONS FOR CALCULATING LOSS
        ########################################################
        self.reservoir['logits'] = tf.reshape(self._inference(self.batch_img), (-1, 10)) # from Nx1x1x10 to Nx10
                
        self.losses = self._createLoss(name='loss')
    
    
    def _createLoss(self, name=None, reuse=tf.AUTO_REUSE):
        ###################################################
        # LOSS CALCULATION
        ##################################################
        with tf.variable_scope(name, reuse=reuse):
            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.reservoir['logits'])    
            loss = tf.reduce_mean(xentropy)
                        
        return loss
            
    
    def optimizer(self):
        
        ####################################################
        # OPTIMIZERS (i.e., Adam or RMSProp, etc) SETTING
        #####################################################
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = opt.minimize(self.losses)
                        
        ############################################################
        # create session 
        ############################################################
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        # when multiple GPUs - gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
                                
        return train_op, self.losses, self.losses, self.reservoir['logits']
    
    
class VGG16(Network):
    
    def __init__(self, cfg):
        
        self.batch_img = tf.placeholder(dtype=tf.uint8) # CIFAR-10 image batch
        self.batch_lab = tf.placeholder(dtype=tf.uint8) # CIFAR-10 label batch
        self.labels = tf.cast(self.batch_lab, tf.int32)
        self.btrain = tf.placeholder(dtype=tf.bool)
        self.breset = tf.placeholder(dtype=tf.bool)
        self.buse_drop = tf.placeholder(dtype=tf.bool)
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.sum_losses = tf.placeholder(dtype=tf.float32) # summary losses
        
        self.reservoir = {} #empty dict for loss related tensors
        
        self.cfg = cfg
                
        super(VGG16, self).__init__()
        
    def __inference(self, tin): #VGG16 without BN
        
        inputs = tf.cast(self.batch_img, tf.float32)/255. -0.5 # make input as float32 and in the range (-0.5, 0.5)
        padding='REFLECT'
        act='ReLu' #provides better performance than ReLu
        init='he_normal'
        
        #conv(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sname=None):
        (self.feed(inputs)
             .convact(lfilter_shape=(3,3,3,64), spadding=padding, sinitializer=init, sactivation=act, sname='conv1-1')
             .convact(lfilter_shape=(3,3,64,64), spadding=padding, sinitializer=init, sactivation=act, sname='conv1-2')
             .maxpool(sname='pool1') #16x16
             .convact(lfilter_shape=(3,3,64,128), spadding=padding, sinitializer=init, sactivation=act, sname='conv2-1')
             .convact(lfilter_shape=(3,3,128,128), spadding=padding, sinitializer=init, sactivation=act, sname='conv2-2')
             .maxpool(sname='pool2') #8x8
             .convact(lfilter_shape=(3,3,128,256), spadding=padding, sinitializer=init, sactivation=act, sname='conv3-1')
             .convact(lfilter_shape=(3,3,256,256), spadding=padding, sinitializer=init, sactivation=act, sname='conv3-2')
             .convact(lfilter_shape=(3,3,256,256), spadding=padding, sinitializer=init, sactivation=act, sname='conv3-3')
             .maxpool(sname='pool3') #4x4
             .convact(lfilter_shape=(3,3,256,512), spadding=padding, sinitializer=init, sactivation=act, sname='conv4-1')
             .convact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, sname='conv4-2')
             .convact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, sname='conv4-3')
             .maxpool(sname='pool4') #2x2
             .convact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, sname='conv5-1')
             .convact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, sname='conv5-2')
             .convact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, sname='conv5-3')
             #.maxpool(sname='pool5') #1x1
             .convact(lfilter_shape=(2,2,512,512), spadding='VALID', sinitializer=init, sactivation=act, sname='FC1')
             .conv(lfilter_shape=(1,1,512,10), spadding='VALID', buse_bias=True, sinitializer=init, sname='out'))
        
        return self.terminals[0]
    
    
    def _inference(self, tin): #VGG16 with BN
        
        inputs = tf.cast(self.batch_img, tf.float32)/255. -0.5 # make input as float32 and in the range (-0.5, 0.5)
        padding='REFLECT'
        act='ReLu' #provides better performance than ReLu
        init='he_normal'
        
        #convbnact(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sactivation='ReLu', brst=False, btrain=True, sname=None):
        (self.feed(inputs)
             .convbnact(lfilter_shape=(3,3,3,64), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv1-1')
             .convbnact(lfilter_shape=(3,3,64,64), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv1-2')
             .maxpool(sname='pool1') #16x16
             .convbnact(lfilter_shape=(3,3,64,128), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv2-1')
             .convbnact(lfilter_shape=(3,3,128,128), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv2-2')
             .maxpool(sname='pool2') #8x8
             .convbnact(lfilter_shape=(3,3,128,256), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv3-1')
             .convbnact(lfilter_shape=(3,3,256,256), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv3-2')
             .convbnact(lfilter_shape=(3,3,256,256), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv3-3')
             .maxpool(sname='pool3') #4x4
             .convbnact(lfilter_shape=(3,3,256,512), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv4-1')
             .convbnact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv4-2')
             .convbnact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv4-3')
             .maxpool(sname='pool4') #2x2
             .convbnact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv5-1')
             .convbnact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv5-2')
             .convbnact(lfilter_shape=(3,3,512,512), spadding=padding, sinitializer=init, sactivation=act, brst=self.breset, btrain=self.btrain, sname='conv5-3')
             #.maxpool(sname='pool5') #1x1
             .flatten()
             .densebnact(iin_nodes=2048, iout_nodes=512, buse_bias=False, sactivation=act, sinitializer=init, brst=self.breset, btrain=self.btrain, sname='FC1')
             .dropout(frate=0.2, buse_drop=self.buse_drop, sname='drop1')
             .dense(iin_nodes=512, iout_nodes=10, sinitializer=init, sname='out'))
        
        return self.terminals[0]
    
            
    def _build(self):
        
        ########################################################
        # NEEDED OPERATIONS FOR CALCULATING LOSS
        ########################################################
        self.reservoir['logits'] = tf.reshape(self._inference(self.batch_img), (-1, 10)) # from Nx1x1x10 to Nx10
                
        self.losses = self._createLoss(name='loss')
    
    
    def _createLoss(self, name=None, reuse=tf.AUTO_REUSE):
        ###################################################
        # LOSS CALCULATION
        ##################################################
        with tf.variable_scope(name, reuse=reuse):
            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.reservoir['logits'])    
            loss = tf.reduce_mean(xentropy)
                        
        return loss
            
    
    def optimizer(self):
        
        ###############################################
        #  WEIGHT DECAY (exclude Batch Norm Params)
        ##############################################
        t_var = tf.trainable_variables()
        w_var = [var for var in t_var if not('_bn' in var.name)]
        w_l2 = tf.add_n([tf.nn.l2_loss(var) for var in w_var])
        loss_to_opt = self.losses + self.cfg.weight_decay * w_l2
        
        ####################################################
        # OPTIMIZERS (i.e., Adam or RMSProp, etc) SETTING
        #####################################################
        opt = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate)
        train_op = opt.minimize(loss_to_opt)
                        
        ############################################################
        # create session 
        ############################################################
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        # when multiple GPUs - gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        
        #########################################################
        # check-point processing
        #########################################################
        self.saver = tf.train.Saver()
        ckpt_loc = self.cfg.ckpt_dir
        self.ckpt_name = os.path.join(ckpt_loc, 'VGG16')
        
        ckpt = tf.train.get_checkpoint_state(ckpt_loc)
        if ckpt and ckpt.model_checkpoint_path:
            import re
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.start_epoch = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0)) + 1
            print("---------------------------------------------------------")
            print(" Success to load checkpoint - {}".format(ckpt_name))
            print(" Session starts at epoch - {}".format(self.start_epoch))
            print("---------------------------------------------------------")
        else:
            if not os.path.exists(ckpt_loc):
                os.makedirs(ckpt_loc)
            self.start_epoch = 0
            print("**********************************************************")
            print("  [*] Failed to find a checkpoint - Start from the first")
            print(" Session starts at epoch - {}".format(self.start_epoch))
            print("**********************************************************")
        
        #################################################
        # Summary and Summary Writer
        #################################################
        wd_loss = tf.summary.scalar("WD_Loss", self.sum_losses[0])
        ce_loss = tf.summary.scalar("CE_Loss", self.sum_losses[1])
        accuracy = tf.summary.scalar("Accuracy", self.sum_losses[2])
                
        self.summary_vgg = tf.summary.merge((wd_loss, ce_loss, accuracy))
        
        self.writer = tf.summary.FileWriter(self.cfg.log_dir, self.sess.graph)
        
        return train_op, loss_to_opt, self.losses, self.reservoir['logits'], self.summary_vgg
    
    
    def save(self, global_step):
                
        self.saver.save(self.sess, self.ckpt_name, global_step)
        print('\n The checkpoint has been created, epoch: {} \n'.format(global_step))


class ResNet34(Network):
    
    def __init__(self, cfg):
        
        self.batch_img = tf.placeholder(dtype=tf.uint8) # CIFAR-10 image batch
        self.batch_lab = tf.placeholder(dtype=tf.uint8) # CIFAR-10 label batch
        self.labels = tf.cast(self.batch_lab, tf.int32)
        self.btrain = tf.placeholder(dtype=tf.bool)
        self.buse_drop = tf.placeholder(dtype=tf.bool)
        self.breset = tf.placeholder(dtype=tf.bool)
        self.learning_rate = tf.placeholder(dtype=tf.float32)
        
        self.reservoir = {} #empty dict for loss related tensors
        
        self.cfg = cfg
                
        super(ResNet34, self).__init__()
        
       
    def _inference(self, tin): #RES34 without BN
        
        inputs = tf.cast(self.batch_img, tf.float32)/255. - 0.5 # make input as float32 and in the range (-1, 1)
        padding='REFLECT'
        act='ReLu' # type of activation ?
        init='he_normal'
        stype='SHORT'
        #conv(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sname=None):
        #resblk(self, tinputs, iCin, iCout, istride=1, buse_bias=False, sinit='he_normal', stype='SHORT', sactivation='ReLu', sname=None):
        (self.feed(inputs)
             .convact(lfilter_shape=(3,3,3,64), istride=2, spadding=padding, sinitializer=init, sactivation=act, sname='conv1') #16x16
             #.maxpool(ipool_size=3, istrides=2, spadding=padding, sname='pool1') #16x16
             .resblk(64, 64, sinit=init, stype=stype, sactivation=act, sname='resblk2-1')
             .resblk(64, 64, sinit=init, stype=stype, sactivation=act, sname='resblk2-2')
             .resblk(64, 64, sinit=init, stype=stype, sactivation=act, sname='resblk2-3')
             .resblk(64, 128, istride=2, sinit=init, stype=stype, sactivation=act, sname='resblk3-1') #8x8
             .resblk(128, 128, sinit=init, stype=stype, sactivation=act, sname='resblk3-2')
             .resblk(128, 128, sinit=init, stype=stype, sactivation=act, sname='resblk3-3')
             .resblk(128, 128, sinit=init, stype=stype, sactivation=act, sname='resblk3-4')
             .resblk(128, 256, istride=2, sinit=init, stype=stype, sactivation=act, sname='resblk4-1') #4x4
             .resblk(256, 256, sinit=init, stype=stype, sactivation=act, sname='resblk4-2')
             .resblk(256, 256, sinit=init, stype=stype, sactivation=act, sname='resblk4-3')
             .resblk(256, 256, sinit=init, stype=stype, sactivation=act, sname='resblk4-4')
             .resblk(256, 256, sinit=init, stype=stype, sactivation=act, sname='resblk4-5')
             .resblk(256, 256, sinit=init, stype=stype, sactivation=act, sname='resblk4-6')
             .resblk(256, 512, istride=2, sinit=init, stype=stype, sactivation=act, sname='resblk5-1') #2x2
             .resblk(512, 512, sinit=init, stype=stype, sactivation=act, sname='resblk5-2')
             .resblk(512, 512, sinit=init, stype=stype, sactivation=act, sname='resblk5-3')
             #dense(iin_nodes=10, iout_nodes=10, btrain=True, bbrst=False, buse_bias=True, buse_bn=False, bbn_first=True, sactivation='ReLu', sinitializer='he_uniform', sname=None):
             .flatten()
             #.dropout(frate=0.2, buse_drop=self.buse_drop, sname='drop1')
             .denseact(iin_nodes=2048, iout_nodes=512, buse_bias=False, sactivation=act, sinitializer=init, sname='FC1')
             #.dropout(frate=0.2, buse_drop=self.buse_drop, sname='drop2')
             .dense(iin_nodes=512, iout_nodes=10, sinitializer=init, sname='out'))
        ##dense(self, tinputs, iin_nodes=10, iout_nodes=10, buse_bias=True, sinitializer='he_uniform', sname=None):
        return self.terminals[0]
               
                                
    def _build(self):
        
        ########################################################
        # NEEDED OPERATIONS FOR CALCULATING LOSS
        ########################################################
        self.reservoir['logits'] = tf.reshape(self._inference(self.batch_img), (-1, 10)) # from Nx1x1x10 to Nx10
                
        self.losses = self._createLoss(name='loss')
    
    
    def _createLoss(self, name=None, reuse=tf.AUTO_REUSE):
        ###################################################
        # LOSS CALCULATION
        ##################################################
        with tf.variable_scope(name, reuse=reuse):
            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.reservoir['logits'])    
            loss = tf.reduce_mean(xentropy)
                        
        return loss
            
    
    def optimizer(self):
        
        ###############################################
        #  WEIGHT DECAY (exclude Batch Norm Params)
        ##############################################
        t_var = tf.trainable_variables()
        w_var = [var for var in t_var if not('_bn' in var.name)]
        w_l2 = tf.add_n([tf.nn.l2_loss(var) for var in w_var])
        loss_to_opt = self.losses + self.cfg.weight_decay * w_l2
        
        ####################################################
        # OPTIMIZERS (i.e., Adam or RMSProp, etc) SETTING
        #####################################################
        #opt = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate, beta1=0.5, beta2=0.9)
        opt = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate)
        #opt = tf.train.MomentumOptimizer(self.cfg.learning_rate, momentum=0.9)
        train_op = opt.minimize(loss_to_opt)
                        
        ############################################################
        # create session 
        ############################################################
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        # when multiple GPUs - gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
                                
        return train_op, loss_to_opt, self.losses, self.reservoir['logits']
        
