import network_layers_v2 as nly
from network_model import NetworkModel


class ResNet34(NetworkModel):
    def __init__(self, img_classifier):
        super().__init__(img_classifier)
        self.net_params = img_classifier.net_params

    def _inference(self, inputs):
        input_layer = nly.InputLayer()
        if self.net_params.use_batch_norm:
            conv1_layer = nly.ConvBnActLayer(self.net_params)
            dense1_layer = nly.DenseBnActLayer(self.net_params)
        else:
            conv1_layer = nly.ConvActLayer(self.net_params)
            dense1_layer = nly.DenseActLayer(self.net_params)
        dense_out_layer = nly.DenseLayer(self.net_params)
        addon = nly.AddOnLayer(self.net_params)
        maxpool = nly.MaxPoolLayer(self.net_params)
        res_block = nly.ResBlockLayer(self.net_params, type='SHORT')

        input_layer.op(inputs)
        conv1_layer.op(lfilter_shape=(3, 3, 3, 64), istride=2, sname='conv1')
        #maxpool.op(ipool_size=3, istrides=2, sname='maxpool1', )
        res_block.op(iCin=64, iCout=64, istride=1, sname='res11')
        res_block.op(iCin=64, iCout=64, istride=1, sname='res12')
        res_block.op(iCin=64, iCout=64, istride=1, sname='res13')

        res_block.op(iCin=64, iCout=128, istride=2, sname='res21')
        res_block.op(iCin=128, iCout=128, istride=1, sname='res22')
        res_block.op(iCin=128, iCout=128, istride=1, sname='res23')
        res_block.op(iCin=128, iCout=128, istride=1, sname='res24')

        res_block.op(iCin=128, iCout=256, istride=2, sname='res31')
        res_block.op(iCin=256, iCout=256, istride=1, sname='res32')
        res_block.op(iCin=256, iCout=256, istride=1, sname='res33')
        res_block.op(iCin=256, iCout=256, istride=1, sname='res34')
        res_block.op(iCin=256, iCout=256, istride=1, sname='res35')
        res_block.op(iCin=256, iCout=256, istride=1, sname='res36')

        res_block.op(iCin=256, iCout=512, istride=2, sname='res41')
        res_block.op(iCin=512, iCout=512, istride=1, sname='res42')
        res_block.op(iCin=512, iCout=512, istride=1, sname='res43')

        addon.flatten()
        dense1_layer.op(iin_nodes=2048, iout_nodes=512, buse_bias=False, sname='fconn1')
        addon.dropout(sname='drop1')
        dense_out_layer.op(iin_nodes=512, iout_nodes=10, sname='out')

        return dense_out_layer.retrieve_from_terminal()


class ResNet50(NetworkModel):
    def __init__(self, img_classifier):
        super().__init__(img_classifier)
        self.net_params = img_classifier.net_params

    def _inference(self, tin):
        # shape
        sh_n, sh_h, sh_w, sh_c = self.t_batch_img.shape
        # make input as float32 and in the range (0, 1)
        inputs = tf.cast(tf.reshape(self.t_batch_img,
                                    (-1, sh_h, sh_w, sh_c)), tf.float32) / 255

        input_layer = nly.InputLayer()
        if self.net_params.use_batch_norm:
            conv1_layer = nly.ConvBnActLayer(self.net_params)
            dense1_layer = nly.DenseBnActLayer(self.net_params)
        else:
            conv1_layer = nly.ConvActLayer(self.net_params)
            dense1_layer = nly.DenseActLayer(self.net_params)
        dense_out_layer = nly.DenseLayer(self.net_params)
        maxpool = nly.MaxPoolLayer(self.net_params)
        addon = nly.AddOnLayer(self.net_params)
        res_block = nly.ResBlockLayer(self.net_params, type='LONG')

        input_layer.op(inputs)
        conv1_layer.op(lfilter_shape=(3, 3, 3, 64), istride=2, sname='conv1')
        #maxpool.op(ipool_size=3, istrides=2, sname='maxpool1', )
        res_block.op(iCin=64, iCout=256, istride=1, sname='res11')
        res_block.op(iCin=256, iCout=256, istride=1, sname='res12')
        res_block.op(iCin=256, iCout=256, istride=1, sname='res13')

        res_block.op(iCin=256, iCout=512, istride=2, sname='res21')
        res_block.op(iCin=512, iCout=512, istride=1, sname='res22')
        res_block.op(iCin=512, iCout=512, istride=1, sname='res23')
        res_block.op(iCin=512, iCout=512, istride=1, sname='res24')

        res_block.op(iCin=512, iCout=1024, istride=2, sname='res31')
        res_block.op(iCin=1024, iCout=1024, istride=1, sname='res32')
        res_block.op(iCin=1024, iCout=1024, istride=1, sname='res33')
        res_block.op(iCin=1024, iCout=1024, istride=1, sname='res34')
        res_block.op(iCin=1024, iCout=1024, istride=1, sname='res35')
        res_block.op(iCin=1024, iCout=1024, istride=1, sname='res36')

        res_block.op(iCin=1024, iCout=2048, istride=2, sname='res41')
        res_block.op(iCin=2048, iCout=2048, istride=1, sname='res42')
        res_block.op(iCin=2048, iCout=2048, istride=1, sname='res43')

        addon.flatten()
        dense1_layer.op(iin_nodes=8192, iout_nodes=2048, buse_bias=False, sname='fconn1')
        addon.dropout(sname='drop1')
        dense_out_layer.op(iin_nodes=2048, iout_nodes=10, sname='out')

        return dense_out_layer.retrieve_from_terminal()
