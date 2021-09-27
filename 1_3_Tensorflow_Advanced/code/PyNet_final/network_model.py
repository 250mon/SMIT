import tensorflow.compat.v1 as tf
import network_levels as nlv
import util_datasets
import pdb


class PyNet():
    def __init__(self, isp_net):
        self.isp_net = isp_net
        self.settings = isp_net.settings
        # getting global network parameters
        self.net_params = isp_net.net_params
        self.vgg19 = util_datasets.NeuralLoss()
        # placeholders for inputs
        self.t_batch_img = tf.placeholder(dtype=tf.uint16, name='input_images')
        self.t_batch_lab = tf.placeholder(dtype=tf.uint8, name='image_labels')
        # placeholders of net_params
        self.net_params.ph_learning_rate = tf.placeholder(dtype=tf.float32,
                                                          name='learning_rate')
        # for tensorboard summary creation of loss and accuracy
        self.sum_losses = tf.placeholder(dtype=tf.float32, name='sum_losses')

        # for debugging
        self.t_probes = None
        ########################################################
        # Train and Evaluation Operations
        ########################################################
        # make inputs as float32 and in the range (-1, 1)
        # N x H x W x C
        self.t_inputs = tf.cast(self.t_batch_img, tf.float32) / 511.5 - 1
        # 5 x N x H x W x C
        self.t_level_labs = self._create_level_labels()
        # infer the outputs of the levels
        # 5 x N x H x W x C
        self.t_outputs = self._inference()
        # a tensor of loss functions of each level
        self.t_loss_ops = self._create_losses()
        # build optimizer; target op is loss_wd
        self.train_ops = self._create_optimizers()

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
        lvl_mgr = nlv.LevelManager(self.isp_net)
        lvl_mgr.build(self.t_inputs)

        t_outputs = {
            'lv0': lvl_mgr.level0.retrieve_from_terminal(),
            'lv1': lvl_mgr.level1.retrieve_from_terminal(),
            'lv2': lvl_mgr.level2.retrieve_from_terminal(),
            'lv3': lvl_mgr.level3.retrieve_from_terminal(),
            'lv4': lvl_mgr.level4.retrieve_from_terminal(),
            'lv5': lvl_mgr.level5.retrieve_from_terminal(),
        }

        # for debugging
        # self.t_probes = lvl_mgr.tensor_probes
        return t_outputs

    def _create_level_labels(self):
        # original label: 448 x 448
        t_level_labs = {
            'lv0': tf.cast(self.t_batch_lab, tf.float32) / 127.5 - 1,
        }
        size = 224
        for i in range(5):
            level = 'lv' + str(i + 1)
            level_lab = tf.image.resize_bicubic(self.t_batch_lab, (size, size),
                                                align_corners=True,
                                                name=level + '_resize_bicubic')
            level_lab = tf.cast(level_lab, tf.float32) / 127.5 - 1
            t_level_labs[level] = level_lab
            size //= 2
        return t_level_labs

    def _mse_loss(self, level):
        return tf.reduce_mean(tf.square(self.t_outputs[level] -
                                        self.t_level_labs[level]),
                              name=level + '_loss')

    def _neural_loss(self, level):
        return self.vgg19.get_layer_loss(self.t_outputs[level],
                                         self.t_level_labs[level])

    def _ssim(self, level):
        # shape: broadcast(N, N) => (N,)
        t_ssim = tf.image.ssim(self.t_outputs[level] + 1,
                               self.t_level_labs[level] + 1,
                               max_val=2.0)
        # returns mean of traced values
        return tf.reduce_mean(t_ssim)

    def _ssim_loss(self, level):
        return 1.0 - self._ssim(level)

    def _psnr(self, level):
        # N x 1 tensor
        t_psnr = tf.image.psnr(self.t_outputs[level] + 1,
                               self.t_level_labs[level] + 1,
                               max_val=2.0)
        # returns a mean
        return tf.reduce_mean(t_psnr)

    def _create_losses(self):
        losses = {
            'lv5':
            self._mse_loss('lv5') * 100,
            'lv4':
            self._mse_loss('lv4') * 100,
            'lv3':
            self._mse_loss('lv3') * 100 + self._neural_loss('lv3'),
            'lv2':
            self._mse_loss('lv2') * 100 + self._neural_loss('lv2'),
            'lv1':
            self._mse_loss('lv1') * 50 + self._neural_loss('lv1'),
            'lv0':
            self._mse_loss('lv0') * 20 + self._neural_loss('lv0') +
            self._ssim_loss('lv0') * 20,
            'eval_ssim':
            self._ssim('lv0'),
            'eval_psnr':
            self._psnr('lv0'),
        }
        return losses

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

    def _create_optimizers(self):
        train_ops = {}
        for lv_name, loss_op in self.t_loss_ops.items():
            train_ops[lv_name] = self._create_optimizer( loss_op, lv_name + '_Adam')
        return train_ops

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
        lvl1_loss = tf.summary.scalar("Lv1_Loss", self.sum_losses[0])
        lvl2_loss = tf.summary.scalar("Lv2_Loss", self.sum_losses[1])
        lvl3_loss = tf.summary.scalar("Lv3_Loss", self.sum_losses[2])
        lvl4_loss = tf.summary.scalar("Lv4_Loss", self.sum_losses[3])
        lvl5_loss = tf.summary.scalar("Lv5_Loss", self.sum_losses[4])
        accuracy = tf.summary.scalar("Accuracy", self.sum_losses[5])

        return tf.summary.merge((lvl1_loss, lvl2_loss, lvl3_loss, lvl4_loss, lvl5_loss, accuracy))
