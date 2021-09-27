import numpy as np
import tensorflow.compat.v1 as tf
import network_layers as nls
import pdb


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
        convbnact_layer.op(lfilter_shape=(3, 3, 3, 32), istride=2, buse_bias=True, sactivation='ReLu', sname='midbr_conv1_1')
        convbnact_layer.op(lfilter_shape=(3, 3, 32, 32), buse_bias=True, sactivation='ReLu', sname='midbr_conv1_2')
        convbnact_layer.op(lfilter_shape=(3, 3, 32, 64), buse_bias=True, sactivation='ReLu', sname='midbr_conv1_3')
        maxpool_layer.op(ipool_size=3, istride=2, sname='midbr_maxpool')
        res_blk_layer.op(iCin=64, iCout=256, istride=1, sactivation='ReLu', sname='midbr_conv2_1')
        res_blk_layer.op(iCin=256, iCout=256, istride=1, sactivation='ReLu', sname='midbr_conv2_2')
        res_blk_layer.op(iCin=256, iCout=256, istride=1, sactivation='ReLu', sname='midbr_conv2_3')
        res_blk_layer.op(iCin=256, iCout=512, istride=2, sactivation='ReLu', sname='midbr_conv3_1')
        return self.retrieve_from_terminal()


class LowBranch(nls.Layer):
    def __init__(self, ic_net, term_name='low_br'):
        super().__init__(term_name)
        self.ic_net = ic_net
        self.net_params = ic_net.net_params
        self.term_name = term_name

    def build(self, tinputs):
        convbnact_layer = nls.ConvBnActLayer(self.net_params, term_name=self.term_name)
        res_blk_layer = nls.ResBlockLayer(self.net_params, term_name=self.term_name, type='LONG')
        pyramidpool_layer = nls.PyramidPoolLayer(self.net_params, term_name=self.term_name)
        addon_layer = nls.AddOnLayer(self.net_params, term_name=self.term_name)

        self.push_to_terminal(tinputs)
        addon_layer.resize_images(np.multiply(tinputs.shape[1:3], 0.5).astype(np.int32), sname='lowbr_downsample_by_2')
        res_blk_layer.op(iCin=512, iCout=512, istride=1, sactivation='ReLu', sname='lowbr_conv3_2')
        res_blk_layer.op(iCin=512, iCout=512, istride=1, sactivation='ReLu', sname='lowbr_conv3_3')
        res_blk_layer.op(iCin=512, iCout=512, istride=1, sactivation='ReLu', sname='lowbr_conv3_4')
        res_blk_layer.op(iCin=512, iCout=1024, istride=1, idilations=2, sactivation='ReLu', sname='lowbr_conv4_1')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=4, sactivation='ReLu', sname='lowbr_conv4_2')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=6, sactivation='ReLu', sname='lowbr_conv4_3')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=2, sactivation='ReLu', sname='lowbr_conv4_4')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=4, sactivation='ReLu', sname='lowbr_conv4_5')
        res_blk_layer.op(iCin=1024, iCout=1024, istride=1, idilations=6, sactivation='ReLu', sname='lowbr_conv4_6')
        res_blk_layer.op(iCin=1024, iCout=2048, istride=1, idilations=2, sactivation='ReLu', sname='lowbr_conv5_1')
        res_blk_layer.op(iCin=2048, iCout=2048, istride=1, idilations=4, sactivation='ReLu', sname='lowbr_conv5_2')
        res_blk_layer.op(iCin=2048, iCout=2048, istride=1, idilations=6, sactivation='ReLu', sname='lowbr_conv5_3')
        pyramidpool_layer.op(sname='lowbr_pyramidpool')
        convbnact_layer.op(lfilter_shape=(1, 1, 2048, 256), buse_bias=True, sactivation='ReLu', sname='11_lowbr_dim_reduction')
        return self.retrieve_from_terminal()


class HighBranch(nls.Layer):
    def __init__(self, ic_net, term_name='high_br'):
        super().__init__(term_name)
        self.ic_net = ic_net
        self.net_params = ic_net.net_params
        self.term_name = term_name

    def build(self, tinputs):
        convbnact_layer = nls.ConvBnActLayer(self.net_params, term_name=self.term_name)

        self.push_to_terminal(tinputs)
        convbnact_layer.op(lfilter_shape=(3, 3, 3, 32), istride=2, buse_bias=True, sactivation='ReLu', sname='highbr_conv1_1')
        convbnact_layer.op(lfilter_shape=(3, 3, 32, 32), istride=2, buse_bias=True, sactivation='ReLu', sname='highbr_conv1_2')
        convbnact_layer.op(lfilter_shape=(3, 3, 32, 64), istride=2, buse_bias=True, sactivation='ReLu', sname='highbr_conv1_3')
        return self.retrieve_from_terminal()


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
        sname = term_name

        # for F1
        self.push_to_terminal(tinputs_f1)
        f1_hw_dim = tinputs_f1.shape[1:3]
        f1_c_dim = tinputs_f1.shape[3]
        t_f1_tap = addon_layer.resize_images(np.multiply(f1_hw_dim, 2), sname=sname+'_upsample_by_2')
        conv_layer.op(lfilter_shape=(3, 3, f1_c_dim, 128), idilations=2, buse_bias=True, sname=sname+'_F1_conv')
        t_f1_out = addon_layer.batch_norm(iCin=128, smode='CONV', sname=sname+'_F1_bn')

        # for F1 loss
        self.push_to_terminal(t_f1_tap)
        t_f1_classified = conv_layer.op(lfilter_shape=(1, 1, f1_c_dim, self.net_params.class_num), buse_bias=True, sname='11_'+sname+'_F1_class_conv')

        # for F2
        self.push_to_terminal(tinputs_f2)
        f2_c_dim = tinputs_f2.shape[3]
        conv_layer.op(lfilter_shape=(1, 1, f2_c_dim, 128), buse_bias=True, sname='11_'+sname+'_F2_conv')
        t_f2_out = addon_layer.batch_norm(iCin=128, smode='CONV', sname=sname+'_F2_bn')

        # sum and act
        t_summed = tf.math.add(t_f1_out, t_f2_out, name=sname+'_sum')
        self.push_to_terminal(t_summed)
        t_act_out = self.addon_layer.activation(sactivation, sname=sname+'_summed_act')

        return t_f1_classified, t_act_out


