import tensorflow.compat.v1 as tf
import network_layers as nls
from tensorflow import keras

# input: 1/2 sized image
# output: 1/16 sized image
class MidBranch(nls.Layer):
    def __init__(self, ic_net, term_name='mid_br'):
        super().__init__(term_name)
        self.ic_net = ic_net
        self.net_params = ic_net.net_params
        self.term_name = term_name

    def build(self, tinputs):
        convbnact_layer = nls.ConvBnActLayer(self.net_params, term_name=self.term_name)
        res_blk_layer = nls.ResBlockLayer(self.net_params, term_name=self.term_name, type='LONG')
        maxpool_layer = nls.MaxPoolLayer(self.net_params, term_name=self.term_name)

        self.push_to_terminal(tinputs)
        convbnact_layer.op(lfilter_shape=(3, 3, 3, 32), istride=2, buse_bias=True, sname='midbr_conv1_1')
        convbnact_layer.op(lfilter_shape=(3, 3, 32, 32), buse_bias=True, sname='midbr_conv1_2')
        convbnact_layer.op(lfilter_shape=(3, 3, 32, 64), buse_bias=True, sname='midbr_conv1_3')
        maxpool_layer.op(ipool_size=3, istride=2, sname='midbr_maxpool')
        res_blk_layer.op(iCin=64, iCout=256, istride=1, sname='midbr_conv2_1')
        res_blk_layer.op(iCin=256, iCout=256, istride=1, sname='midbr_conv2_2')
        res_blk_layer.op(iCin=256, iCout=256, istride=1, sname='midbr_conv2_3')
        res_blk_layer.op(iCin=256, iCout=512, istride=2, sname='midbr_conv3_1')
        return self.retrieve_from_terminal()

# input: 1/16 sized image
# output: 1/32 sized image
class LowBranch(nls.Layer):
    def __init__(self, ic_net, term_name='low_br'):
        super().__init__(term_name)
        self.ic_net = ic_net
        self.net_params = ic_net.net_params
        self.term_name = term_name

    def build(self, tinputs):
        convbnact_layer = nls.ConvBnActLayer(self.net_params, term_name=self.term_name)
        res_blk_layer = nls.ResBlockLayer(self.net_params, term_name=self.term_name, type='LONG')
        pyramidpool_layer = nls.PyramidPoolLayer2(self.net_params, term_name=self.term_name)
        addon_layer = nls.AddOnLayer(self.net_params, term_name=self.term_name)

        self.push_to_terminal(tinputs)
        hw_size = tf.cast(tf.shape(tinputs)[1:3], tf.float32)
        addon_layer.resize_images(tf.cast(tf.multiply(hw_size, 0.5), tf.int32), sname='lowbr_downsample_by_2')
        res_blk_layer.op(iCin=512, iCout=512, istride=1, sname='lowbr_conv3_2')
        res_blk_layer.op(iCin=512, iCout=512, istride=1, sname='lowbr_conv3_3')
        res_blk_layer.op(iCin=512, iCout=512, istride=1, sname='lowbr_conv3_4')
        res_blk_layer.op(iCin=512, iCout=1024, istride=1, idilations=2, sname='lowbr_conv4_1')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=4, sname='lowbr_conv4_2')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=6, sname='lowbr_conv4_3')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=2, sname='lowbr_conv4_4')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=4, sname='lowbr_conv4_5')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=6, sname='lowbr_conv4_6')
        res_blk_layer.op(iCin=1024, iCout=2048, istride=1, idilations=2, sname='lowbr_conv5_1')
        res_blk_layer.op(iCin=2048, iCout=2048, istride=1, idilations=4, sname='lowbr_conv5_2')
        res_blk_layer.op(iCin=2048, iCout=2048, istride=1, idilations=6, sname='lowbr_conv5_3')
        pyramidpool_layer.op(sname='lowbr_pyramidpool')
        convbnact_layer.op(lfilter_shape=(1, 1, 2048, 256), buse_bias=True, sname='11_lowbr_dim_reduction')
        return self.retrieve_from_terminal()

class HighBranchPre(nls.Layer):
    def __init__(self, ic_net, term_name='high_br_pre'):
        super().__init__(term_name)
        self.ic_net = ic_net
        self.net_params = ic_net.net_params
        self.term_name = term_name

    def build(self, tinputs):
        convbnact_layer = nls.ConvBnActLayer(self.net_params, term_name=self.term_name)

        self.push_to_terminal(tinputs)
        convbnact_layer.op(lfilter_shape=(3, 3, 3, 32), istride=2, buse_bias=True, sname=self.term_name+'_conv1_1')
        convbnact_layer.op(lfilter_shape=(3, 3, 32, 32), istride=2, buse_bias=True, sname=self.term_name+'_conv1_2')
        convbnact_layer.op(lfilter_shape=(3, 3, 32, 64), istride=2, buse_bias=True, sname=self.term_name+'_conv1_3')
        return self.retrieve_from_terminal()

class HighBranchPost(nls.Layer):
    def __init__(self, ic_net, term_name='high_br_post'):
        super().__init__(term_name)
        self.ic_net = ic_net
        self.net_params = ic_net.net_params
        self.term_name = term_name

    def build(self, tinputs):
        addon_layer = nls.AddOnLayer(self.net_params, term_name=self.term_name)
        conv_layer = nls.ConvLayer(self.net_params, term_name=self.term_name)

        self.push_to_terminal(tinputs)
        # upsampling by 2 with bilinear interpolation
        t_tap = addon_layer.resize_images(tf.multiply(tf.shape(tinputs)[1:3], 2), sname=self.term_name+'_upsample_by_2')
        t_high_segmented = conv_layer.op(lfilter_shape=(1, 1, tinputs.shape[3], self.net_params.class_num), buse_bias=True, sname='11_'+self.term_name+'_F1_class_conv')
        # upsampling by 4 with bilinear interpolation
        addon_layer.resize_images(tf.multiply(tf.shape(t_tap)[1:3], 4), sname=self.term_name+'_upsample_by_4')
        # The final output retrieved has 19 channels
        return t_high_segmented, self.retrieve_from_terminal()

# using Keras
# For upsampling by 2, use a transposed convolutional layer instead of bilinear interpolation
class HighBranchPost2(nls.Layer):
    def __init__(self, ic_net, term_name='high_br_post'):
        super().__init__(term_name)
        self.ic_net = ic_net
        self.net_params = ic_net.net_params
        self.term_name = term_name

    def build(self, tinputs):
        addon_layer = nls.AddOnLayer(self.net_params, term_name=self.term_name)
        conv_layer = nls.ConvLayer(self.net_params, term_name=self.term_name)

        # self.push_to_terminal(tinputs)
        # t_tap = addon_layer.resize_images(tf.multiply(tf.shape(tinputs)[1:3], 2), sname=self.term_name+'_upsample_by_2')
        # t_high_segmented = conv_layer.op(lfilter_shape=(1, 1, tinputs.shape[3], self.net_params.class_num), buse_bias=True, sname='11_'+self.term_name+'_F1_class_conv')
        # addon_layer.resize_images(tf.multiply(tf.shape(t_tap)[1:3], 4), sname=self.term_name+'_upsample_by_4')

        # upsampling by 2 with transposed convolution
        t_input = keras.layers.Input(tensor=tinputs)
        t_1_4 = keras.layers.Conv2DTranspose(t_input.shape[3], (2, 2), strides=(2, 2), activation="relu", padding="same")(t_input)
        t_high_segmented = keras.layers.Conv2D(self.net_params.class_num, (1, 1), activation="relu", padding="same")(t_1_4)
        t_1_1 = keras.layers.Conv2DTranspose(t_1_4.shape[3], (4, 4), strides=(4, 4), activation="relu", padding="same")(t_1_4)
        # The final output retrieved has 19 channels
        return t_high_segmented, t_1_1

class CFFModule(nls.Layer):
    def __init__(self, ic_net, term_name='CFF'):
        super().__init__(term_name)
        self.ic_net = ic_net
        self.net_params = ic_net.net_params
        self.term_name = term_name

    # tinputs: F1 as lowbr_tap up until this point, F2 as inputs which is actually midbr output
    def build(self, tinputs_f1, tinputs_f2):
        addon_layer = nls.AddOnLayer(self.net_params, term_name=self.term_name)
        conv_layer = nls.ConvLayer(self.net_params, term_name=self.term_name)
        sname = self.term_name

        # for F1
        self.push_to_terminal(tinputs_f1)
        f1_hw_dim = tf.shape(tinputs_f1)[1:3]
        # hack; hw is supposed to be 45 (720/16).
        # but, after a couple of stride convs, the size ends up being 22.5
        # and eventually, it becomes 44 by upsampling and leads to an error
        # if f1_hw_dim[0] == 22:
        #     f1_upsampled_size = np.multiply(f1_hw_dim, 2) + 1
        # else:
        #     f1_upsampled_size = np.multiply(f1_hw_dim, 2)

        f1_upsampled_size = tf.cond(tf.math.equal(f1_hw_dim[0], 22),
                                    lambda: tf.math.multiply(f1_hw_dim, 2)+1,
                                    lambda: tf.math.multiply(f1_hw_dim, 2))

        # f1_c_dim = tf.shape(tinputs_f1)[3]
        f1_c_dim = tinputs_f1.shape[3]
        t_f1_tap = addon_layer.resize_images(f1_upsampled_size, sname=sname+'_upsample_by_2')
        conv_layer.op(lfilter_shape=(3, 3, f1_c_dim, 128), idilations=2, buse_bias=True, sname=sname+'_F1_conv')
        t_f1_out = addon_layer.batch_norm(iCin=128, smode='CONV', sname=sname+'_F1_bn')

        # for F1 loss
        self.push_to_terminal(t_f1_tap)
        t_f1_segmented = conv_layer.op(lfilter_shape=(1, 1, f1_c_dim, self.net_params.class_num), buse_bias=True, sname='11_'+sname+'_F1_class_conv')

        # for F2
        self.push_to_terminal(tinputs_f2)
        # f2_c_dim = tf.shape(tinputs_f2)[3]
        f2_c_dim = tinputs_f2.shape[3]
        conv_layer.op(lfilter_shape=(1, 1, f2_c_dim, 128), buse_bias=True, sname='11_'+sname+'_F2_conv')
        t_f2_out = addon_layer.batch_norm(iCin=128, smode='CONV', sname=sname+'_F2_bn')

        # sum and act
        t_summed = tf.math.add(t_f1_out, t_f2_out, name=sname+'_sum')
        self.push_to_terminal(t_summed)
        t_act_out = addon_layer.activation(sname=sname+'_summed_act')

        return t_f1_segmented, t_act_out


