import tensorflow.compat.v1 as tf
from abc import ABC, abstractmethod
import os


class NetworkModel(ABC):
    def __init__(self, img_classifier):
        self.settings = img_classifier.settings
        # getting global network parameters
        self.net_params = img_classifier.net_params
        # placeholders for inputs
        self.t_batch_img = tf.placeholder(dtype=tf.uint8, name='input_images')
        self.t_batch_lab = tf.placeholder(dtype=tf.uint8, name='image_labels')
        # placeholders of net_params
        self.net_params.ph_learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.net_params.ph_bn_reset = tf.placeholder(dtype=tf.bool, name='b_bn_reset')
        self.net_params.ph_bn_train = tf.placeholder(dtype=tf.bool, name='b_bn_train')
        self.net_params.ph_use_drop = tf.placeholder(dtype=tf.bool, name='b_drop_out')
        # merge
        self.sum_losses = tf.placeholder(dtype=tf.float32, name='sum_losses')
        ########################################################
        # Train and Evaluation Operations
        ########################################################
        # type conversion of labels; uint8 to int32
        # labels (N,) 
        self.t_labels = tf.cast(self.t_batch_lab, tf.int32)
        # make input as float32 and in the range (-0.5, 0.5)
        inputs = tf.cast(self.t_batch_img, tf.float32)/255. - 0.5
        # logits tensor (N, 10) from the inference op
        self.t_logits = tf.reshape(self._inference(inputs), (-1, 10)) # from Nx1x1x10 to Nx10
        # predicted labels tensor (N,); final output node (for excavating graph)
        self.t_predicted_labels = tf.argmax(self.t_logits, axis=1, name='predicted_labels', output_type=tf.int32)

        # build loss calculator (tensor)
        self.loss = self._create_loss(name='loss')
        # apply weight decay term to the loss
        self.loss_wd = self._apply_weight_decay()

        # build optimizer; target op is loss_wd
        self.train_op = self._create_optimizer(self.loss_wd)
        # build session
        self.session = self._create_session()

        # summaries
        self.summaries = self._create_summary()
        # summary writer
        self.summary_writer = tf.summary.FileWriter(self.settings.tb_log_dir, self.session.graph)

    @abstractmethod
    def _inference(self, tin):
        pass

    def _apply_weight_decay(self):
        ###############################################
        #  WEIGHT DECAY (exclude Batch Norm Params)
        ##############################################
        if self.net_params.use_weight_decay == False:
            return self.loss

        t_var = tf.trainable_variables()
        w_var = [var for var in t_var if not('_bn' in var.name)]
        w_l2 = tf.add_n([tf.nn.l2_loss(var) for var in w_var])
        loss_wd = self.loss + self.net_params.weight_decay * w_l2
        return loss_wd

    def _create_loss(self, name=None, reuse=tf.AUTO_REUSE):
        ########################################################
        # DEFINTION OF LOSS CALCULATION
        ########################################################
        with tf.variable_scope(name, reuse=reuse):
        # sparse means that labels is converted to one hot vector to be (N, 10)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.t_labels, logits=self.t_logits)
            loss = tf.reduce_mean(xentropy)
        return loss

    def _create_session(self):
        ########################################################
        # CREATE SESSION
        ########################################################
        # when multiple GPUs - gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        # create tensorflow session
        sess = tf.Session(config=config)
        # initialize all tensorflow variables (assign initial values)
        sess.run(tf.global_variables_initializer())
        return sess

    def _create_optimizer(self, op_to_minimize):
        ########################################################
        # OPTIMIZERS (i.e., Adam or RMSProp, etc) SETTING
        ########################################################
        opt = tf.train.AdamOptimizer(
                learning_rate=self.net_params.ph_learning_rate,
                beta1=self.net_params.adam_beta1,
                beta2=self.net_params.adam_beta2
                )
        train_op = opt.minimize(op_to_minimize)
        return train_op

    def _create_summary(self):
        wd_loss = tf.summary.scalar("WD_Loss", self.sum_losses[0])
        ce_loss = tf.summary.scalar("CE_Loss", self.sum_losses[1])
        accuracy = tf.summary.scalar("Accuracy", self.sum_losses[2])

        return tf.summary.merge((wd_loss, ce_loss, accuracy))
