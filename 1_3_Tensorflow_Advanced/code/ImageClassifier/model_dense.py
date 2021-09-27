import network_layers_v2 as nly
from network_model import NetworkModel


class DenseNet(NetworkModel):
    def __init__(self, img_classifier):
        super().__init__(img_classifier)
        self.net_params = img_classifier.net_params

    def _inference(self, inputs):
        input_layer = nly.InputLayer()
        dense_layer = nly.DenseLayer(self.net_params)
        denseact_layer = nly.DenseActLayer(self.net_params)
        addon = nly.AddOnLayer(self.net_params)

        input_layer.op(inputs)
        addon.flatten()
        denseact_layer.op(iin_nodes=784, iout_nodes=256, sname='hidden-1')
        denseact_layer.op(iin_nodes=256, iout_nodes=512, sname='hidden-2')
        denseact_layer.op(iin_nodes=512, iout_nodes=256, sname='hidden-3')
        # num_label output without activation
        dense_layer.op(iin_nodes=256, iout_nodes=10, sname='out')

        return dense_layer.retrieve_from_terminal()
