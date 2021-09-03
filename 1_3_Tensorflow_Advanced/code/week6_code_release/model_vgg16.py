import tensorflow.compat.v1 as tf
from network import NetworkLayers

tf.disable_eager_execution()
    
class ConvNet(NetworkLayers):
    
    def __init__(self, cfg):
        
        self.batch_img = tf.placeholder(dtype=tf.uint8, shape=(None, 32, 32, 3))
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
        # shape
        sh_n, sh_h, sh_w, sh_c = self.batch_img.shape
        # make input as float32 and in the range (0, 1)
        inputs = tf.cast(tf.reshape(self.batch_img, (-1, sh_h, sh_w, sh_c)), tf.float32)/255
        (self.feed(inputs)
             .conv(lfilter_shape=(3,3,3,64), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv1')
             .conv(lfilter_shape=(3,3,64,64), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv2')
             .maxpool(sname='maxpool1')
             .conv(lfilter_shape=(3,3,64,128), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv3')
             .conv(lfilter_shape=(3,3,128,128), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv4')
             .maxpool(sname='maxpool2')
             .conv(lfilter_shape=(3,3,128,256), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv5')
             .conv(lfilter_shape=(3,3,256,256), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv6')
             .conv(lfilter_shape=(3,3,256,256), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv7')
             .maxpool(sname='maxpool3')
             .conv(lfilter_shape=(3,3,256,512), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv8')
             .conv(lfilter_shape=(3,3,512,512), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv9')
             .conv(lfilter_shape=(3,3,512,512), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv10')
             .maxpool(sname='maxpool4')
             .conv(lfilter_shape=(2,2,512,512), lstrides=(1,1,1,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv11')
             # When the input gets through the conv filter of the same dimension with a VALID padding, it is meant to be fully coneected to the output
             .conv(lfilter_shape=(2,2,512,10), lstrides=(1,1,1,1), spadding='VALID', buse_bias=True, sactivation='None', sinitializer='he_normal', sname='out'))
        
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


        
        
