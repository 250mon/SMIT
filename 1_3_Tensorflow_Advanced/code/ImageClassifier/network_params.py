

class NetParams:
    def __init__(self, img_classifier):
        self.img_classifier = img_classifier
        self.settings = img_classifier.settings 
        # global parameters that are shared in common by all the network components
        self._activation = 'ReLu'
        self._initializer = 'he_normal'
        # for conv net
        self._padding = 'REFLECT'
        # for drop out
        self._drop_rate = 0.2
        # for weight decay
        self._weight_decay = 0.0005
        # for adam opt
        self._adam_beta1 = 0.9
        self._adam_beta2 = 0.999
        # for grid search of regularizers
        self._use_drop = True
        self._use_batch_norm = True
        self._use_data_aug = True
        self._use_weight_decay = True
        # network paramter placeholders that will defined in NetworkModel
        self.ph_learning_rate = None
        self.ph_bn_reset = None
        self.ph_bn_train = None
        self.ph_use_drop = None
        
    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, new_act):
        self._activation = new_act

    @property
    def initializer(self):
        return self._initializer

    @initializer.setter
    def initializer(self, new_init):
        self._initializer = new_init

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, new_padding):
        self._padding = new_padding

    @property
    def use_drop(self):
        return self._use_drop

    @use_drop.setter
    def use_drop(self, new_use_drop):
        self._use_drop = new_use_drop

    @property
    def drop_rate(self):
        return self._drop_rate

    @drop_rate.setter
    def drop_rate(self, new_drop_rate):
        self._drop_rate = new_drop_rate

    @property
    def weight_decay(self):
        return self._weight_decay

    @weight_decay.setter
    def weight_decay(self, new_weight_decay):
        self._weight_decay = new_weight_decay

    @property
    def use_batch_norm(self):
        return self._use_batch_norm

    @use_batch_norm.setter
    def use_batch_norm(self, new_use_batch_norm):
        self._use_batch_norm = new_use_batch_norm

    @property
    def use_data_aug(self):
        return self._use_data_aug

    @use_data_aug.setter
    def use_data_aug(self, new_use_data_aug):
        self._use_data_aug = new_use_data_aug

    @property
    def use_weight_decay(self):
        return self._use_weight_decay

    @use_weight_decay.setter
    def use_weight_decay(self, new_use_weight_decay):
        self._use_weight_decay = new_use_weight_decay

    @property
    def adam_beta1(self):
        return self._adam_beta1

    @adam_beta1.setter
    def adam_beta1(self, new_adam_beta1):
        self._adam_beta1 = new_adam_beta1

    @property
    def adam_beta2(self):
        return self._adam_beta2

    @adam_beta2.setter
    def adam_beta2(self, new_adam_beta2):
        self._adam_beta2 = new_adam_beta2

