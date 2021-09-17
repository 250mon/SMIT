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
        
        if self.cfg.net_type == 'Resnet34':
            (self.feed(inputs)
                 .conv(lfilter_shape=(3,3,3,64), lstrides=(1,2,2,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv1')
                 #.maxpool(ipool_size=3, istrides=2, sname='maxpool1', )
                 .resblock(iCin=64, iCout=64, lstrides=(1,1,1,1), ctype='SHORT', sname='res11')
                 .resblock(iCin=64, iCout=64, lstrides=(1,1,1,1), ctype='SHORT', sname='res12')
                 .resblock(iCin=64, iCout=64, lstrides=(1,1,1,1), ctype='SHORT', sname='res13')
                 .resblock(iCin=64, iCout=128, lstrides=(1,2,2,1), ctype='SHORT', sname='res21')
                 .resblock(iCin=128, iCout=128, lstrides=(1,1,1,1), ctype='SHORT', sname='res22')
                 .resblock(iCin=128, iCout=128, lstrides=(1,1,1,1), ctype='SHORT', sname='res23')
                 .resblock(iCin=128, iCout=128, lstrides=(1,1,1,1), ctype='SHORT', sname='res24')
                 .resblock(iCin=128, iCout=256, lstrides=(1,2,2,1), ctype='SHORT', sname='res31')
                 .resblock(iCin=256, iCout=256, lstrides=(1,1,1,1), ctype='SHORT', sname='res32')
                 .resblock(iCin=256, iCout=256, lstrides=(1,1,1,1), ctype='SHORT', sname='res33')
                 .resblock(iCin=256, iCout=256, lstrides=(1,1,1,1), ctype='SHORT', sname='res34')
                 .resblock(iCin=256, iCout=256, lstrides=(1,1,1,1), ctype='SHORT', sname='res35')
                 .resblock(iCin=256, iCout=256, lstrides=(1,1,1,1), ctype='SHORT', sname='res36')
                 .resblock(iCin=256, iCout=512, lstrides=(1,2,2,1), ctype='SHORT', sname='res41')
                 .resblock(iCin=512, iCout=512, lstrides=(1,1,1,1), ctype='SHORT', sname='res42')
                 .resblock(iCin=512, iCout=512, lstrides=(1,1,1,1), ctype='SHORT', sname='res43')
                 # When the input gets through the conv filter of the same dimension with a VALID padding, it is meant to be fully coneected to the output
                 .conv(lfilter_shape=(2,2,512,512), lstrides=(1,1,1,1), spadding='VALID', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv2')
                 .conv(lfilter_shape=(1,1,512,10), lstrides=(1,1,1,1), spadding='VALID', buse_bias=True, sactivation='None', sinitializer='he_normal', sname='out'))
        
        # Resnet50
        else:
            (self.feed(inputs)
                 .conv(lfilter_shape=(3,3,3,64), lstrides=(1,2,2,1), spadding='SAME', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv1')
                 #.maxpool(ipool_size=3, istrides=2, sname='maxpool1', )
                 .resblock(iCin=64, iCout=256, lstrides=(1,1,1,1), ctype='LONG', sname='res11')
                 .resblock(iCin=256, iCout=256, lstrides=(1,1,1,1), ctype='LONG', sname='res12')
                 .resblock(iCin=256, iCout=256, lstrides=(1,1,1,1), ctype='LONG', sname='res13')
                 .resblock(iCin=256, iCout=512, lstrides=(1,2,2,1), ctype='LONG', sname='res21')
                 .resblock(iCin=512, iCout=512, lstrides=(1,1,1,1), ctype='LONG', sname='res22')
                 .resblock(iCin=512, iCout=512, lstrides=(1,1,1,1), ctype='LONG', sname='res23')
                 .resblock(iCin=512, iCout=512, lstrides=(1,1,1,1), ctype='LONG', sname='res24')
                 .resblock(iCin=512, iCout=1024, lstrides=(1,2,2,1), ctype='LONG', sname='res31')
                 .resblock(iCin=1024, iCout=1024, lstrides=(1,1,1,1), ctype='LONG', sname='res32')
                 .resblock(iCin=1024, iCout=1024, lstrides=(1,1,1,1), ctype='LONG', sname='res33')
                 .resblock(iCin=1024, iCout=1024, lstrides=(1,1,1,1), ctype='LONG', sname='res34')
                 .resblock(iCin=1024, iCout=1024, lstrides=(1,1,1,1), ctype='LONG', sname='res35')
                 .resblock(iCin=1024, iCout=1024, lstrides=(1,1,1,1), ctype='LONG', sname='res36')
                 .resblock(iCin=1024, iCout=2048, lstrides=(1,2,2,1), ctype='LONG', sname='res41')
                 .resblock(iCin=2048, iCout=2048, lstrides=(1,1,1,1), ctype='LONG', sname='res42')
                 .resblock(iCin=2048, iCout=2048, lstrides=(1,1,1,1), ctype='LONG', sname='res43')
                 # When the input gets through the conv filter of the same dimension with a VALID padding, it is meant to be fully coneected to the output
                 .conv(lfilter_shape=(2,2,2048,2048), lstrides=(1,1,1,1), spadding='VALID', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname='conv2')
                 .conv(lfilter_shape=(1,1,2048,10), lstrides=(1,1,1,1), spadding='VALID', buse_bias=True, sactivation='None', sinitializer='he_normal', sname='out'))

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
