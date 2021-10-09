import tensorflow.compat.v1 as tf
import tensorflow.bitwise as tw
import numpy as np
import network_branches as nbr


class NetModel():
    def __init__(self, ic_net):
        self.ic_net = ic_net
        self.settings = ic_net.settings
        # getting global network parameters
        self.net_params = ic_net.net_params
        # placeholders for inputs
        self.t_batch_img = tf.placeholder(dtype=tf.uint8, name='input_images')
        self.t_batch_lab = tf.placeholder(dtype=tf.uint8, name='image_labels')
        # placeholders of net_params
        self.net_params.ph_learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        self.net_params.ph_bn_reset = tf.placeholder(dtype=tf.bool, name='b_bn_reset')
        self.net_params.ph_bn_train = tf.placeholder(dtype=tf.bool, name='b_bn_train')
        self.net_params.ph_use_drop = tf.placeholder(dtype=tf.bool, name='b_drop_out')
        # for tensorboard summary creation of loss and accuracy
        self.ph_summary = tf.placeholder(dtype=tf.float32, name='summary')

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
        # show the predicted labels
        self.t_pred_labels = tf.expand_dims(tf.argmax(self.t_outputs, axis=3, name='predicted_labels', output_type=tf.int32), -1)
        # calc miou
        self.t_gt_labels_32 = tf.cast(self.t_batch_lab, tf.int32)
        self.miou, self.t_class_ious = self._calc_iou(self.t_pred_labels, self.t_gt_labels_32)

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
        highbr_pre = nbr.HighBranchPre(self.ic_net)
        highbr_post = nbr.HighBranchPost(self.ic_net)
        cff1 = nbr.CFFModule(self.ic_net, term_name='cff1')
        cff2 = nbr.CFFModule(self.ic_net, term_name='cff2')
        org_size = self._get_org_hw_size()

        t_midbr_conv_out = midbr.build(self._resize_images(self.t_inputs, (org_size*0.5).astype(np.int32)))
        t_lowbr_out = lowbr.build(t_midbr_conv_out)
        # t_lowbr_pred 1/16
        t_lowbr_pred, t_midbr_out = cff1.build(t_lowbr_out, t_midbr_conv_out)
        lowbr_loss = self._loss(t_lowbr_pred, self.t_level_labs['lowbr'])

        t_highbr_conv_out = highbr_pre.build(self.t_inputs)
        # t_midbr_pred 1/8
        t_midbr_pred, t_highbr_pre_out = cff2.build(t_midbr_out, t_highbr_conv_out)
        midbr_loss = self._loss(t_midbr_pred, self.t_level_labs['midbr'])

        # t_highbr_pred is a classified output (1/4 size, 19 channels)
        # t_outputs is a final output (1 size, 19 channels)
        t_highbr_pred, t_outputs = highbr_post.build(t_highbr_pre_out)
        highbr_loss = self._loss(t_highbr_pred, self.t_level_labs['highbr'])

        loss = 0.4 * lowbr_loss + 0.4 * midbr_loss + highbr_loss
        return loss, t_outputs

    # image and label size (h x w) are the same
    def _get_org_hw_size(self):
        return np.array(self.ic_net.dataset.TRAIN_SIZE)

    # input size: (h, w)
    def _resize_images(self, t_images, size):
        return tf.image.resize_bilinear(t_images, size, align_corners=True)

    # input size: (h, w)
    def _resize_labels(self, t_labels, size):
        return tf.image.resize_nearest_neighbor(t_labels, size, align_corners=True)

    def _create_branch_labels(self):
        # original label: 720 x 720
        factors = {
                'highbr': 0.25,
                'midbr': 0.125,
                'lowbr': 0.0625,
                }
        org_size = self._get_org_hw_size()
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
        # serialize t_gt
        t_gt_serial = tf.reshape(t_gt, (-1,))
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
            learning_rate=self.net_params.ph_learning_rate,
            beta1=self.net_params.adam_beta1,
            beta2=self.net_params.adam_beta2,
            name=name,
        )
        train_op = opt.minimize(op_to_minimize)
        return train_op

    def _create_summary(self):
        loss = tf.summary.scalar("Loss", self.ph_summary[0])
        loss_wd = tf.summary.scalar("Loss_Wd", self.ph_summary[1])
        miou = tf.summary.scalar("mIoU", self.ph_summary[2])
        class_ious = [tf.summary.scalar(f"Class_IoU{i}", self.ph_summary[i+3]) for i in range(19)]
        return tf.summary.merge((loss, loss_wd, miou, *class_ious))

    # makes 2d confusion matrix
    def _make_confusion_matrix(self, preds, gts):
        # pred 10, gt 9 => 0x090a (gt, pred) pair
        merged_maps = tw.bitwise_or(tw.left_shift(gts, 8), preds)
        # hist indices: 0x0000 ~ 0x1212 (gt:18, pred:18, 4626)
        # hist: 1 dim
        hist = tf.bincount(merged_maps, minlength=0x10000)
        hist_2d = tf.reshape(hist, (256, 256))
        conf_matrix_2d = hist_2d[:19, :19]

        # # nonzero bin indices
        # # nonzero_indices: 2 dim
        # nonzero_indices = tf.where(tf.math.not_equal(hist, 0))
        #
        # # nonzero_values: 2 dim -> 1 dim
        # nonzero_values_2d = tf.gather(hist, nonzero_indices)
        # nonzero_values_1d = tf.squeeze(nonzero_values_2d)
        # # indices(1d or 2d), output_shape(1d), values(1d)
        # conf_matrix_1d = tf.sparse_to_dense(nonzero_indices,
        #                                     ((256 * 256),),
        #                                     nonzero_values_1d,
        #                                     0)
        # conf_matrix_2d = tf.reshape(conf_matrix_1d, (256, 256))
        return conf_matrix_2d

    # calculate Mean IoU
    def _calc_iou(self, preds, gts):
        conf_matrix = self._make_confusion_matrix(preds, gts)
        # sum elements for each row
        row_sum = tf.squeeze(tf.reduce_sum(conf_matrix, axis=1))
        # sum elements for each col
        col_sum = tf.squeeze(tf.reduce_sum(conf_matrix, axis=0))
        # number of classes appeared in current gt label
        # gt_class_num = tf.cast(tf.count_nonzero(row_sum), dtype=tf.float64)
        # diagonal elements (the number of True Positive for all classes)
        diag = tf.squeeze(tf.diag_part(conf_matrix))
        union = row_sum + col_sum - diag
        # only nonzeros of union
        nonzero_indices = tf.where(tf.not_equal(union, 0))
        union_nonzero_values_2d = tf.gather(union, nonzero_indices)
        union_nonzero_values_1d = tf.squeeze(union_nonzero_values_2d)
        union_nonzero_values_1d = tf.cast(union_nonzero_values_1d, tf.float64)
        diag_nonzero_values_2d = tf.gather(diag, nonzero_indices)
        diag_nonzero_values_1d = tf.squeeze(diag_nonzero_values_2d)
        diag_nonzero_values_1d = tf.cast(diag_nonzero_values_1d, tf.float64)
        union_nonzero_size = tf.cast(tf.size(union_nonzero_values_1d, out_type=tf.int32), dtype=tf.float64)
        # IoUs of each class
        class_ious = tf.math.divide_no_nan(diag_nonzero_values_1d, union_nonzero_values_1d)
        # sum all IoUs and divide the number of classes
        mIoU = tf.math.truediv(tf.reduce_sum(tf.math.truediv(diag_nonzero_values_1d, union_nonzero_values_1d)), union_nonzero_size)
        # mIoU = tf.truediv(tf.reduce_sum(tf.truediv(diag_nonzero_values_1d, union_nonzero_values_1d)), gt_class_num)
        return mIoU, class_ious

