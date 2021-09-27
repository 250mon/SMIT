import network_layers_v2 as nly
from network_model import NetworkModel


class ConvNet(NetworkModel):
    def __init__(self, img_classifier):
        super().__init__(img_classifier)
        self.net_params = img_classifier.net_params

    def _inference(self, inputs):
        input_layer = nly.InputLayer()
        convact_layer = nly.ConvActLayer(self.net_params)
        conv_layer = nly.ConvLayer(self.net_params)
        maxpool = nly.MaxPoolLayer(self.net_params)

        input_layer.op(inputs)
        convact_layer.op(lfilter_shape=(3, 3, 1, 16), istride=1, sname='conv1')
        convact_layer.op(lfilter_shape=(3, 3, 16, 16), istride=1, sname='conv1-1')
        maxpool.op(sname='pool1')
        convact_layer.op(lfilter_shape=(3, 3, 16, 64), istride=1, sname='conv2')
        convact_layer.op(lfilter_shape=(3, 3, 64, 64), istride=1, sname='conv2-1')
        maxpool.op(sname='pool2')
        convact_layer.op(lfilter_shape=(3, 3, 64, 128), istride=1, sname='conv3')
        convact_layer.op(lfilter_shape=(3, 3, 128, 128), istride=1, sname='conv3-1')
        convact_layer.op(lfilter_shape=(3, 3, 128, 256), istride=1, sname='conv4')
        convact_layer.op(lfilter_shape=(3, 3, 256, 256), istride=1, sname='conv4-1')
        conv_layer.op(lfilter_shape=(7, 7, 256, 10), istride=1, spadding='VALID', buse_bias=True, sname='out')

        return conv_layer.retrieve_from_terminal()
