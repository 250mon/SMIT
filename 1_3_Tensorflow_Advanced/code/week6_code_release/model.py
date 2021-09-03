import tensorflow.compat.v1 as tf
from network import NetworkLayers

tf.disable_eager_execution()

class DenseNet(NetworkLayers):
    
    def __init__(self, cfg):
        # None: batch size is not fixed but variable
        self.batch_img = tf.placeholder(dtype=tf.uint8, shape=(None, 784))
        self.batch_lab = tf.placeholder(dtype=tf.uint8, shape=(None))
        self.labels = tf.cast(self.batch_lab, tf.int32)
        
        self.reservoir = {} #empty dict for loss related tensors
        
        self.cfg = cfg
                
        super(DenseNet, self).__init__()
        
        
    def _build(self):
        ########################################################
        # BUILD TRAIN OPERATIONS AND LOSS CALCULATION
        ########################################################
        # build operations (foward pass network)
        self.reservoir['logits'] = self._inference(self.batch_img)
        # build loss calculator
        self.losses = self._createLoss(name='loss')
    

    def _inference(self, tin):
        ########################################################
        # DEFINTION OF TRAIN OPERATIONS (FOWARD PASS NETWORK)
        ########################################################
        # make input as float32 and in the range (0, 1)
        inputs = tf.cast(tin, tf.float32)/255.
        # 3 layer dense net with (512, 256, 512) nodes
        (self.feed(inputs)
             .dense(iin_nodes=784, iout_nodes=256, buse_bias=True, sactivation='ReLu', sinitializer='he_uniform', sname='hidden-1')
             .dense(iin_nodes=256, iout_nodes=512, buse_bias=True, sactivation='ReLu', sinitializer='he_uniform', sname='hidden-2')
             .dense(iin_nodes=512, iout_nodes=256, buse_bias=True, sactivation='ReLu', sinitializer='he_uniform', sname='hidden-3')
             .dense(iin_nodes=256, iout_nodes=10, buse_bias=True, sactivation='None', sinitializer='he_uniform', sname='out')) # num_label output without activation
        
        return self.terminals['main'][0]
        
    
    def _createLoss(self, name=None, reuse=tf.AUTO_REUSE):
        ########################################################
        # DEFINTION OF LOSS CALCULATION
        ########################################################
        with tf.variable_scope(name, reuse=reuse):
            # labels (256,)  reservoir['logits'] (256, 10)
            # sparse means that labels is converted to one hot vector to be (256, 10)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.reservoir['logits'])    
            loss = tf.reduce_mean(xentropy)
                        
        return loss
            
    
    def optimizer(self):
        ########################################################
        # OPTIMIZERS (i.e., Adam or RMSProp, etc) SETTING
        ########################################################
        opt = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate, beta1=0.9, beta2=0.999)
        train_op = opt.minimize(self.losses)
                        
        ########################################################
        # CREATE SESSION
        ########################################################
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        # when multiple GPUs - gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

        # create tensorflow session
        self.sess = tf.Session(config=config)
        # initialize all tensorflow variables (assign initial values)
        self.sess.run(tf.global_variables_initializer())
                                
        return train_op, self.losses, self.reservoir['logits']
    
    
class ConvNet(NetworkLayers):
    
    def __init__(self, cfg):
        
        self.batch_img = tf.placeholder(dtype=tf.uint8, shape=(None, 784))
        self.batch_lab = tf.placeholder(dtype=tf.uint8, shape=(None))
        self.labels = tf.cast(self.batch_lab, tf.int32)
        
        self.reservoir = {} #empty dict for loss related tensors
        
        self.cfg = cfg
                
        super(ConvNet, self).__init__()
        
    def _build(self):
        ########################################################
        # BUILD TRAIN OPERATIONS AND LOSS CALCULATION
        ########################################################
        # build operations (foward pass network)
        self.reservoir['logits'] = tf.reshape(self._inference(self.batch_img), (-1, 10)) # from Nx1x1x10 to Nx10
        # build loss calculator
        self.losses = self._createLoss(name='loss')
    
    def _inference(self, tin):
        # make input as float32 and in the range (0, 1)
        inputs = tf.cast(tf.reshape(self.batch_img, (-1, 28, 28, 1)), tf.float32)/255
        """ Old network """
        # (self.feed(inputs)
             # .conv(lfilter_shape=(5,5,1,16), lstrides=(1,2,2,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv1')
             # .conv(lfilter_shape=(5,5,16,32), lstrides=(1,2,2,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv2')
             # .conv(lfilter_shape=(5,5,32,64), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv3')
             # .conv(lfilter_shape=(7,7,64,10), lstrides=(1,1,1,1), spadding='VALID', buse_bias=True, sactivation='None', sinitializer='he_normal', sname='out'))

        """ New network """
        initializer_str = 'he_normal'
        activation_str = 'ReLu'
        # initializer_str = 'glorot_normal'
        # activation_str = 'LReLu'

        (self.feed(inputs)
             .conv(lfilter_shape=(3,3,1,16), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation=activation_str, sinitializer=initializer_str, sname='conv1')
             .conv(lfilter_shape=(3,3,16,16), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation=activation_str, sinitializer=initializer_str, sname='conv1-1')
             .maxpool(sname='pool1')
             .conv(lfilter_shape=(3,3,16,64), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation=activation_str, sinitializer=initializer_str, sname='conv2')
             .conv(lfilter_shape=(3,3,64,64), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation=activation_str, sinitializer=initializer_str, sname='conv2-1')
             .maxpool(sname='pool2')
             .conv(lfilter_shape=(3,3,64,128), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation=activation_str, sinitializer=initializer_str, sname='conv3')
             .conv(lfilter_shape=(3,3,128,128), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation=activation_str, sinitializer=initializer_str, sname='conv3-1')
             .conv(lfilter_shape=(3,3,128,256), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation=activation_str, sinitializer=initializer_str, sname='conv4')
             .conv(lfilter_shape=(3,3,256,256), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation=activation_str, sinitializer=initializer_str, sname='conv4-1')
             .conv(lfilter_shape=(7,7,256,10), lstrides=(1,1,1,1), spadding='VALID', buse_bias=True, sactivation='None', sinitializer=initializer_str, sname='out'))
        
        
        return self.terminals['main'][0]
    
    def _createLoss(self, name=None, reuse=tf.AUTO_REUSE):
        ########################################################
        # DEFINTION OF LOSS CALCULATION
        ########################################################
        with tf.variable_scope(name, reuse=reuse):
            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.reservoir['logits'])    
            loss = tf.reduce_mean(xentropy)
                        
        return loss
            
    
    def optimizer(self):
        ########################################################
        # OPTIMIZERS (i.e., Adam or RMSProp, etc) SETTING
        ########################################################
        opt = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate)
        train_op = opt.minimize(self.losses)
                        
        ########################################################
        # CREATE SESSION
        ########################################################
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        # when multiple GPUs - gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
                                
        return train_op, self.losses, self.reservoir['logits']


        
        
