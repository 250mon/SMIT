import network_layers_v2 as nly
from network_model import NetworkModel


class VGG16Net(NetworkModel):
    def __init__(self, img_classifier):
        super().__init__(img_classifier)
        self.net_params = img_classifier.net_params

    def _inference(self, inputs):
        input_layer = nly.InputLayer()
        convbnact_layer = nly.ConvBnActLayer(self.net_params)
        densebnact_layer = nly.DenseBnActLayer(self.net_params)
        dense_layer = nly.DenseLayer(self.net_params)
        maxpool = nly.MaxPoolLayer()
        addon = nly.AddOnLayer(self.net_params)

        input_layer.op(inputs)
        convbnact_layer.op(lfilter_shape=(3, 3, 3, 64), istride=1, sname='conv1')
        convbnact_layer.op(lfilter_shape=(3, 3, 64, 64), istride=1, sname='conv2')
        maxpool.op(sname='maxpool1')
        convbnact_layer.op(lfilter_shape=(3, 3, 64, 128), istride=1, sname='conv3')
        convbnact_layer.op(lfilter_shape=(3, 3, 128, 128), istride=1, sname='conv4')
        maxpool.op(sname='maxpool2')
        convbnact_layer.op(lfilter_shape=(3, 3, 128, 256), istride=1, sname='conv5')
        convbnact_layer.op(lfilter_shape=(3, 3, 256, 256), istride=1, sname='conv6')
        convbnact_layer.op(lfilter_shape=(3, 3, 256, 256), istride=1, sname='conv7')
        maxpool.op(sname='maxpool3')
        convbnact_layer.op(lfilter_shape=(3, 3, 256, 512), istride=1, sname='conv8')
        convbnact_layer.op(lfilter_shape=(3, 3, 512, 512), istride=1, sname='conv9')
        convbnact_layer.op(lfilter_shape=(3, 3, 512, 512), istride=1, sname='conv10')
        maxpool.op(sname='maxpool4')
        convbnact_layer.op(lfilter_shape=(2, 2, 512, 512), istride=1, sname='conv11')
        addon.flatten()
        densebnact_layer.op(iin_nodes=2048, iout_nodes=512, buse_bias=False, sname='fconn1')
        addon.dropout(sname='drop1')
        dense_layer.op(iin_nodes=512, iout_nodes=10, sname='out')

        return dense_layer.retrieve_from_terminal()
