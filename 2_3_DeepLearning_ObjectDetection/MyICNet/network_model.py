import tensorflow.compat.v1 as tf
import numpy as np
import network_branches as nbr
import util_datasets
import pdb


class NetModel():
    def __init__(self, ic_net):
        self.ic_net = ic_net
        self.settings = ic_net.settings
        # getting global network parameters
        self.net_params = ic_net.net_params
        self.vgg19 = util_datasets.NeuralLoss()
        # placeholders for inputs
        self.t_batch_img = tf.placeholder(dtype=tf.uint16, name='input_images')
        self.t_batch_lab = tf.placeholder(dtype=tf.uint8, name='image_labels')
        # placeholders of net_params
        self.net_params.learning_rate_ph = tf.placeholder(dtype=tf.float32, name='learning_rate')
        # for tensorboard summary creation of loss and accuracy
        self.summary_ph = tf.placeholder(dtype=tf.float32, name='sum_losses')

        # for debugging
        self.t_probes = None
        ########################################################
        # Train and Evaluation Operations
        ########################################################
        # make inputs as float32 and in the range (0, 1)
        # N x H x W x C
        self.t_inputs = tf.cast(self.t_batch_img, tf.float32) / 255.0
        # 4 x N x H x W x C
        self.t_level_labs = self._create_branch_labels()
        # infer the outputs and losses of the levels
        self.loss, self.t_outputs = self._inference()
        # apply weight decay
        self.loss_wd = self._apply_weight_decay(self.loss)
        # build optimizer; target op is loss_wd
        self.train_ops = self._create_optimizer(self.loss_wd)

        # build session
        self.session = self._create_session()

        # merged summaries
        self.summaries = self._create_summary()
        # summary writer
        self.summary_writer = tf.summary.FileWriter(self.settings.tb_log_dir,
                                                    self.session.graph)

        # # predicted labels tensor (N,); final output node (for excavating graph)
        # self.t_predicted_labels = tf.argmax(self.t_logits, axis=1, name='predicted_labels', output_type=tf.int32)
        # # correct prediction tensor (1 or 0)
        # self.t_correct_prediction = tf.equal(self.t_predicted_labels, self.t_labels)
        # # accuracy scalar
        # self.accuracy = tf.reduce_mean(tf.cast(self.t_correct_prediction, tf.float32))

    def _inference(self):
        lowbr = nbr.LowBranch(self.ic_net)
        midbr = nbr.MidBranch(self.ic_net)
        highbr = nbr.MidBranch(self.ic_net)
        cff1 = nbr.CFFModule(self.ic_net, term_name='cff1')
        cff2 = nbr.CFFModule(self.ic_net, term_name='cff2')

        t_midbr_conv_out = midbr.build(inputi 1/4)
        t_lowbr_out = lowbr.build(t_midbr_conv_out)
        # t_lowbr_pred 1/16
        t_lowbr_pred, t_midbr_out = cff1.build(t_lowbr_out, t_midbr_conv_out)
        lowbr_loss = self._loss(t_lowbr_pred, self.t_level_labs['lowbr'])

        t_highbr_conv_out = highbr.build(input 1/2)
        # t_midbr_pred 1/8
        t_midbr_pred, t_highbr_out = cff2.build(t_midbr_out, t_highbr_conv_out)
        midbr_loss = self._loss(t_midbr_pred, self.t_level_labs['midbr'])

        # t_highbr_pred 1/4
        t_highbr_pred = self._resize_images(t_highbr_out, np.array(t_highbr_out.shape[1:3])*2)
        highbr_loss = self._loss(t_highbr_pred, self.t_level_labs['highbr'])

        # t_main_out 1
        t_main_out = self._resize_images(t_highbr_pred, np.array(t_highbr_out.shape[1:3])*4)

        loss = 0.4 * lowbr_loss + 0.4 * midbr_loss + highbr_loss
        t_outputs = {'highbr': t_highbr_pred, 'main': t_main_out}
        return loss, t_outputs

    # input size: (h, w)
    def _resize_images(t_images, size):
        return tf.image.resize_bilinear(t_images, size, align_corners=True)

    # input size: (h, w)
    def _resize_labels(t_labels, size):
        return tf.image.resize_nearest_neighbor(t_labels, size, align_corners=True)

    def _create_branch_labels(self):
        # original label: 720 x 720
        factors = {
                'highbr': 0.25,
                'midbr': 0.125,
                'lowbr': 0.0625,
                }
        org_size = np.array(self.t_batch_lab.shape[1:3])
        factors_num = np.array(list(factors.values()))
        sizes = np.outer(factors_num, org_size)

        t_br_labels = {
            'main': self.t_batch_lab,
        }
        for br, size in zip(factors.keys(), sizes):
            t_br_labels[br] = self._resize_labels(self.t_batch_lab, size)
        return t_br_labels

    # t_gt: ground truth, t_pred: prediction
    def _loss(self, t_pred, t_gt):
        t_gt_serial = tf.reshape(t_gt, (-1,)) # serialize
        mask = tf.less_equal(t_gt_serial, self.net_params.class_num - 1)
        indices = tf.squeeze(tf.where(mask), 1)

        t_gt_gathered = tf.cast(tf.gather(t_gt_serial, indices), tf.int32)
        t_pred_gathered = tf.gather(tf.reshape(t_pred, (-1, self.net_params.class_num)), indices)

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=t_pred_gathered, labels=t_gt_gathered))

    # weight decay applied to Batch Norm Params and 1x1 conv Params
    def _apply_weight_decay(self, loss):
        if self.net_params.use_weight_decay == False:
            return self.loss
        t_var = tf.trainable_variables()
        w_var = [var for var in t_var if not('_bn' in var.name) and not('11_' in var.name)]
        w_l2 = tf.add_n([tf.nn.l2_loss(var) for var in w_var])
        loss_wd = self.loss + self.net_params.weight_decay * w_l2
        return loss_wd

    def _create_session(self):
        # when multiple GPUs - gpu_options =
        #   tf.GPUOptions(allow_growth=True, allocator_type='BFC', visible_device_list="0,1")
        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC')
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        # create tensorflow session
        sess = tf.Session(config=config)
        # initialize all tensorflow variables (assign initial values)
        sess.run(tf.global_variables_initializer())
        return sess

    def _create_optimizer(self, op_to_minimize, name='Adam'):
        opt = tf.train.AdamOptimizer(
            learning_rate=self.net_params.learning_rate_ph,
            beta1=self.net_params.adam_beta1,
            beta2=self.net_params.adam_beta2,
            name=name,
        )
        train_op = opt.minimize(op_to_minimize)
        return train_op

    def _create_summary(self):
        loss = tf.summary.scalar("Loss", self.summary_ph[0])
        accuracy = tf.summary.scalar("Accuracy", self.summary_ph[1])
        return tf.summary.merge((loss, accuracy))