import network_layers as nls
import pdb


class LevelManager:
    def __init__(self, isp_net):
        self.level5 = Level5(isp_net, self)
        self.level4 = Level4(isp_net, self)
        self.level3 = Level3(isp_net, self)
        self.level2 = Level2(isp_net, self)
        self.level1 = Level1(isp_net, self)
        self.level0 = Level0(isp_net, self)
        # for debugging
        self.tensor_probes = None

    def build(self, inputs):
        return self.level0.build(inputs)


class Level0(nls.Layer):
    def __init__(self, isp_net, level_mgr, term_name='main'):
        super().__init__(term_name)
        self.isp_net = isp_net
        self.level1 = level_mgr.level1
        self.net_params = isp_net.net_params
        self.term_name = term_name

    def build(self, inputs):
        input_layer = nls.InputLayer(term_name=self.term_name)
        convact = nls.ConvActLayer(self.net_params, term_name=self.term_name)

        input_from_lower, _ = self.level1.build(inputs)
        input_layer.op(input_from_lower)
        lv0_down_out = convact.op(lfilter_shape=(3, 3, 8, 3),
                                  sactivation='TanH',
                                  sname='lv0_outconv')

        return lv0_down_out


class Level1(nls.Layer):
    def __init__(self, isp_net, level_mgr, term_name='main1'):
        super().__init__(term_name)
        self.isp_net = isp_net
        self.level2 = level_mgr.level2
        self.net_params = isp_net.net_params
        self.term_name = term_name

    def build(self, inputs):
        input_layer = nls.InputLayer(term_name=self.term_name)
        conv_blk = nls.ConvBlockLayer(self.net_params, binst_norm=True, term_name=self.term_name)
        addon = nls.AddOnLayer(self.net_params, term_name=self.term_name)
        maxpool = nls.MaxPoolLayer(self.net_params, term_name=self.term_name + '_downstream')
        trconv = nls.TrConvLayer(self.net_params, term_name=self.term_name + '_upstream')
        convact = nls.ConvActLayer(self.net_params, term_name=self.term_name)

        input_layer.op(inputs)
        init_out = conv_blk.op(max_size=3, ch_maps=(4, 32), sname='lv1_conv1')

        # downstream input to the lower level
        maxpool.push_to_terminal(init_out)
        down_input = maxpool.op(sname='lv1_maxpool')
        # getting an upstream flow from the lower lelvel as a feedback
        input_from_lower, _ = self.level2.build(down_input)

        # main
        addon.concat([input_from_lower, ], sname='lv1_concat1')
        conv_blk.op(max_size=5, ch_maps=(64, 32), sname='lv1_conv2')
        addon.concat([init_out, ], sname='lv1_concat2')
        conv_blk.op(max_size=7, ch_maps=(96, 32), sname='lv1_conv3')
        branch = conv_blk.op(max_size=9, ch_maps=(96, 32), sname='lv1_conv4')
        conv_blk.op(max_size=9, ch_maps=(128, 32), sname='lv1_conv5')
        branch = addon.concat([branch, ], sname='lv1_concat3')
        conv_blk.op(max_size=9, ch_maps=(256, 32), sname='lv1_conv6')
        branch = addon.concat([branch, ], sname='lv1_concat4')
        conv_blk.op(max_size=9, ch_maps=(384, 32), sname='lv1_conv7')
        addon.concat([branch, ], sname='lv1_concat5')
        conv_blk.op(max_size=7, ch_maps=(512, 32), sname='lv1_conv8')
        addon.concat([init_out, ], sname='lv1_concat6')
        conv_blk.op(max_size=5, ch_maps=(128, 32), sname='lv1_conv9')
        addon.concat([init_out, input_from_lower, ], sname='lv1_concat7')
        conv_blk.op(max_size=3, ch_maps=(128, 32), sname='lv1_conv10')

        # 2 outputs;
        # one for the upstream to the upper level and
        # the ohter for the current level down the stream
        tapping = conv_blk.retrieve_from_terminal()
        trconv.push_to_terminal(tapping)
        lv1_up_out = trconv.op(lfilter_shape=(3, 3, 8, 32), istride=2, sname='lv1_trconv')
        lv1_down_out = convact.op(lfilter_shape=(3, 3, 32, 3), sactivation='TanH', sname='lv1_outconv')

        return lv1_up_out, lv1_down_out


class Level2(nls.Layer):
    def __init__(self, isp_net, level_mgr, term_name='main2'):
        super().__init__(term_name)
        self.isp_net = isp_net
        self.level3 = level_mgr.level3
        self.lvl_mgr = level_mgr
        self.net_params = isp_net.net_params
        self.term_name = term_name

    def build(self, inputs):
        input_layer = nls.InputLayer(term_name=self.term_name)
        conv_blk = nls.ConvBlockLayer(self.net_params, binst_norm=True, term_name=self.term_name)
        addon = nls.AddOnLayer(self.net_params, term_name=self.term_name)
        maxpool = nls.MaxPoolLayer(self.net_params, term_name=self.term_name + '_downstream')
        trconv = nls.TrConvLayer(self.net_params, term_name=self.term_name + '_upstream')
        convact = nls.ConvActLayer(self.net_params, term_name=self.term_name)

        input_layer.op(inputs)
        init_out = conv_blk.op(max_size=3, ch_maps=(32, 64), sname='lv2_conv1')

        # downstream input to the lower level
        maxpool.push_to_terminal(init_out)
        down_input = maxpool.op(sname='lv2_maxpool')
        # getting an upstream flow from the lower lelvel as a feedback
        input_from_lower, _ = self.level3.build(down_input)

        # main
        addon.concat([input_from_lower, ], sname='lv2_concat1')
        conv_blk.op(max_size=5, ch_maps=(128, 64), sname='lv2_conv2')
        branch = addon.concat([init_out, ], sname='lv2_concat2')
        conv_blk.op(max_size=7, ch_maps=(192, 64), sname='lv2_conv3')
        branch = addon.concat([branch, ], sname='lv2_concat3')
        conv_blk.op(max_size=7, ch_maps=(384, 64), sname='lv2_conv4')
        branch = addon.concat([branch, ], sname='lv2_concat4')
        conv_blk.op(max_size=7, ch_maps=(576, 64), sname='lv2_conv5')
        addon.concat([branch, ], sname='lv2_concat5')
        conv_blk.op(max_size=7, ch_maps=(768, 64), sname='lv2_conv6')
        addon.concat([init_out, ], sname='lv2_concat6')
        conv_blk.op(max_size=5, ch_maps=(256, 64), sname='lv2_conv7')
        addon.concat([input_from_lower, ], sname='lv2_concat8')
        conv_blk.op(max_size=3, ch_maps=(192, 64), sname='lv2_conv8')

        # 2 outputs;
        # one for the upstream to the upper level and
        # the ohter for the current level down the stream
        tapping = conv_blk.retrieve_from_terminal()
        trconv.push_to_terminal(tapping)
        # Cout:32, Cin:64
        lv2_up_out = trconv.op(lfilter_shape=(3, 3, 32, 64), istride=2, sname='lv2_trconv')
        lv2_down_out = convact.op(lfilter_shape=(3, 3, 64, 3), sactivation='TanH', sname='lv2_outconv')

        return lv2_up_out, lv2_down_out


class Level3(nls.Layer):
    def __init__(self, isp_net, level_mgr, term_name='main3'):
        super().__init__(term_name)
        self.isp_net = isp_net
        self.level4 = level_mgr.level4
        self.lvl_mgr = level_mgr
        self.net_params = isp_net.net_params
        self.term_name = term_name

    def build(self, inputs):
        input_layer = nls.InputLayer(term_name=self.term_name)
        conv_blk = nls.ConvBlockLayer(self.net_params, binst_norm=True, term_name=self.term_name)
        addon = nls.AddOnLayer(self.net_params, term_name=self.term_name)
        maxpool = nls.MaxPoolLayer(self.net_params, term_name=self.term_name + '_downstream')
        trconv = nls.TrConvLayer(self.net_params, term_name=self.term_name + '_upstream')
        convact = nls.ConvActLayer(self.net_params, term_name=self.term_name)

        input_layer.op(inputs)
        init_out = conv_blk.op(max_size=3, ch_maps=(64, 128), sname='lv3_conv1')

        # downstream input to the lower level
        maxpool.push_to_terminal(init_out)
        down_input = maxpool.op(sname='lv3_maxpool')
        # getting an upstream flow from the lower lelvel as a feedback
        input_from_lower, _ = self.level4.build(down_input)

        # main
        branch = addon.concat([input_from_lower, ], sname='lv3_concat1')
        conv_blk.op(max_size=5, ch_maps=(256, 128), sname='lv3_conv2')
        branch = addon.concat([branch, ], sname='lv3_concat2')
        conv_blk.op(max_size=5, ch_maps=(512, 128), sname='lv3_conv3')
        branch = addon.concat([branch, ], sname='lv3_concat3')
        conv_blk.op(max_size=5, ch_maps=(768, 128), sname='lv3_conv4')
        addon.concat([branch, ], sname='lv3_concat4')
        conv_blk.op(max_size=5, ch_maps=(1024, 128), sname='lv3_conv5')
        addon.concat([init_out, input_from_lower], sname='lv3_concat5')
        conv_blk.op(max_size=3, ch_maps=(512, 128), sname='lv3_conv6')

        # 2 outputs;
        # one for the upstream to the upper level and
        # the ohter for the current level down the stream
        tapping = conv_blk.retrieve_from_terminal()
        trconv.push_to_terminal(tapping)
        # Cout:64, Cin:128
        lv3_up_out = trconv.op(lfilter_shape=(3, 3, 64, 128), istride=2, sname='lv3_trconv')
        lv3_down_out = convact.op(lfilter_shape=(3, 3, 128, 3), sactivation='TanH', sname='lv3_outconv')

        return lv3_up_out, lv3_down_out


class Level4(nls.Layer):
    def __init__(self, isp_net, level_mgr, term_name='main4'):
        super().__init__(term_name)
        self.isp_net = isp_net
        self.level5 = level_mgr.level5
        self.lvl_mgr = level_mgr
        self.net_params = isp_net.net_params
        self.term_name = term_name

    def build(self, inputs):
        input_layer = nls.InputLayer(term_name=self.term_name)
        conv_blk = nls.ConvBlockLayer(self.net_params, binst_norm=True, term_name=self.term_name)
        addon = nls.AddOnLayer(self.net_params, term_name=self.term_name)
        maxpool = nls.MaxPoolLayer(self.net_params, term_name=self.term_name + '_downstream')
        trconv = nls.TrConvLayer(self.net_params, term_name=self.term_name + '_upstream')
        convact = nls.ConvActLayer(self.net_params, term_name=self.term_name)

        input_layer.op(inputs)
        init_out = conv_blk.op(max_size=3, ch_maps=(128, 256), sname='lv4_conv1')

        # downstream input to the lower level
        maxpool.push_to_terminal(init_out)
        down_input = maxpool.op(sname='lv4_maxpool')
        # getting an upstream flow from the lower lelvel as a feedback
        input_from_lower, _ = self.level5.build(down_input)

        # main
        addon.concat([input_from_lower, ], sname='lv4_concat1')
        branch = conv_blk.op(max_size=3, ch_maps=(512, 256), sname='lv4_conv2')
        conv_blk.op(max_size=3, ch_maps=(256, 256), sname='lv4_conv3')
        branch = addon.add(branch, sname='lv4_add1')
        conv_blk.op(max_size=3, ch_maps=(256, 256), sname='lv4_conv4')
        branch = addon.add(branch, sname='lv4_add2')
        conv_blk.op(max_size=3, ch_maps=(256, 256), sname='lv4_conv5')
        addon.concat([input_from_lower, ], sname='lv4_concat2')
        conv_blk.op(max_size=3, ch_maps=(512, 256), sname='lv4_conv6')

        # 2 outputs;
        # one for the upstream to the upper level and
        # the ohter for the current level down the stream
        tapping = conv_blk.retrieve_from_terminal()
        trconv.push_to_terminal(tapping)
        # Cout:128, Cin:256
        lv4_up_out = trconv.op(lfilter_shape=(3, 3, 128, 256), istride=2, sname='lv4_trconv')
        lv4_down_out = convact.op(lfilter_shape=(3, 3, 256, 3), sactivation='TanH', sname='lv4_outconv')

        return lv4_up_out, lv4_down_out


class Level5(nls.Layer):
    def __init__(self, isp_net, level_mgr, term_name='main5'):
        super().__init__(term_name)
        self.isp_net = isp_net
        self.lvl_mgr = level_mgr
        self.net_params = isp_net.net_params
        self.term_name = term_name

    def build(self, inputs):
        input_layer = nls.InputLayer(term_name=self.term_name)
        conv_blk = nls.ConvBlockLayer(self.net_params, binst_norm=True, term_name=self.term_name)
        addon = nls.AddOnLayer(self.net_params, term_name=self.term_name)
        trconv = nls.TrConvLayer(self.net_params, term_name=self.term_name + '_upstream')
        convact = nls.ConvActLayer(self.net_params, term_name=self.term_name)

        input_layer.op(inputs)
        branch = conv_blk.op(max_size=3, ch_maps=(256, 512), sname='lv5_conv1')
        conv_blk.op(max_size=3, ch_maps=(512, 512), sname='lv5_conv2')
        branch = addon.add(branch, sname='lv5_add1')
        conv_blk.op(max_size=3, ch_maps=(512, 512), sname='lv5_conv3')
        addon.add(branch, sname='lv5_add2')
        conv_blk.op(max_size=3, ch_maps=(512, 512), sname='lv5_conv4')

        tapping = conv_blk.retrieve_from_terminal()
        trconv.push_to_terminal(tapping)
        # Cout:256, Cin:512
        lv5_up_out = trconv.op(lfilter_shape=(3, 3, 256, 512), istride=2, sname='lv5_trconv')
        lv5_down_out = convact.op(lfilter_shape=(3, 3, 512, 3), sactivation='TanH', sname='lv5_outconv')

        return lv5_up_out, lv5_down_out
