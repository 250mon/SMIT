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
        inputs = tf.cast(tin, tf.float32)/127.5 - 1. # make input as float32 and in the range (-1, 1)
        # 3 layer dense net with (512, 256, 512) nodes
        (self.feed(inputs)
             .dense(iin_nodes=784, iout_nodes=256, buse_bias=True, sactivation='ReLu', sinitializer='he_uniform', sname='hidden-1')
             .dense(iin_nodes=256, iout_nodes=512, buse_bias=True, sactivation='ReLu', sinitializer='he_uniform', sname='hidden-2')
             .dense(iin_nodes=512, iout_nodes=256, buse_bias=True, sactivation='ReLu', sinitializer='he_uniform', sname='hidden-3')
             .dense(iin_nodes=256, iout_nodes=10, buse_bias=True, sactivation='None', sinitializer='he_uniform', sname='out')) # num_label output without activation
        
        return self.terminals[0]
        
    
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
        opt = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate)
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


