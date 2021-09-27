import tensorflow.compat.v1 as tf


class CommonFunc:
    @staticmethod
    def get_initializer(sninitializer):
        initializers = {
            'glorot_normal': tf.initializers.glorot_normal(),
            'glorot_uniform': tf.initializers.glorot_uniform(),
            'he_normal': tf.initializers.he_normal(),
            'he_uniform': tf.initializers.he_uniform(),
        }
        # defualt he_uniform
        return initializers.get(sninitializer, tf.initializers.he_uniform())


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class _TerminalHandler:
    def __init__(self):
        self.terminals = {}
        self.create_terminal('main')

    def create_terminal(self, term_name):
        self.terminals.setdefault(term_name, [])

    def pull_terminal(self, term_name):
        return self.terminals[term_name]

    def feed_terminal(self, tinput, term_name):
        self.terminals[term_name] = []
        self.terminals[term_name].append(tinput)


class Layer:
    # the default terminal is main,
    # later it can be changed by creating or changing
    def __init__(self, term_name='main'):
        self.term_handler = _TerminalHandler()
        self.term_name = term_name

    def create_terminal(self, term_name):
        self.term_handler.create_terminal(term_name)
        self.term_name = term_name
        return self

    def change_terminal(self, term_name):
        self.term_name = term_name
        return self

    # retrieve_from_terminal() doesn't remove the old data from the terminal
    # only push_to_termal() can push the old data out of the terminal
    def retrieve_from_terminal(self, term_name=None):
        if term_name is None:
            term = self.term_handler.pull_terminal(self.term_name)[0]
        else:
            term = self.term_handler.pull_terminal(term_name)[0]
        return term

    def push_to_terminal(self, data):
        self.term_handler.feed_terminal(data, self.term_name)


# network model must have one input layer at the beginning
class InputLayer(Layer):
    def __init__(self, term_name='main'):
        super().__init__(term_name)

    def op(self, tinput):
        self.push_to_terminal(tinput)

    """
    Parameters of dense network
    * tinputs:        input tensor
    * iin_nodes:      number of input nodes
    * iout_nodes:     number of output nodes
    * buse_bias:      whether or not to use bias
    * sactivation:    activation function
    * sinitializer:   initializer function
    * sname:          node name
    """


class DenseLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params

    def op(self,
           iin_nodes=10,
           iout_nodes=10,
           buse_bias=True,
           sname=None,
           *args,
           **kwargs):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            # netparam as default
            sinitializer = kwargs.get('sinitializer',
                                      self.net_params.initializer)

            # INITIALIZERS
            init_fn = CommonFunc.get_initializer(sinitializer)

            # WEIGHT VARIABLE
            weights = tf.get_variable(name='weights',
                                      shape=(iin_nodes, iout_nodes),
                                      initializer=init_fn)
            # (N x in_nodes) multiply (in_nodes x out_nodes) = (N x out_nodes)
            x = tf.matmul(tinputs, weights)
            # BIAS
            if buse_bias:
                bias = tf.get_variable(
                    name='bias',
                    shape=(iout_nodes),
                    initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)

            self.push_to_terminal(x)

            return x

    """
    Parameters of convolution layer
    * tinputs:       input tensor(NxHxWxC); convolution done for the last 3 dim
    * filters:       shape of 4D tensor (filter_H x filter_W x in_C x out_C)
    * istrides:      int that is 1, 2, or 4
    * padding:       'SAME' for zero padding or 'VALID' for no padding or ...
    * buse_bias:     whether or not to use bias
    * sactivation:   activation function
    * sinitializer:  initializer function
    * sname:         node name
    """


class ConvLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params

    def op(self,
           lfilter_shape=(3, 3, 1, 1),
           istride=1,
           buse_bias=False,
           sname=None,
           *args,
           **kwargs):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            # netparam as default
            spadding = kwargs.get('spadding', self.net_params.padding)
            sinitializer = kwargs.get('sinitializer',
                                      self.net_params.initializer)
            # INITIALIZERS
            init_fn = CommonFunc.get_initializer(sinitializer)
            # Kernel
            kernels = tf.get_variable(name='kernel',
                                      shape=lfilter_shape,
                                      initializer=init_fn)
            # Padding
            if spadding == 'REFLECT':
                pad1 = tf.cast(tf.subtract(lfilter_shape[0:2], 1) / 2,
                               dtype=tf.int32)
                pad2 = tf.cast(tf.div(lfilter_shape[0:2], 2), dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0], pad2[0]], [pad1[1], pad2[1]],
                            [0, 0]]
                tinputs = tf.pad(tinputs, pad_size, 'REFLECT')
                spadding = 'VALID'
            # Convolution
            x = tf.nn.conv2d(tinputs,
                             kernels,
                             strides=(1, istride, istride, 1),
                             padding=spadding)
            if buse_bias:
                bias = tf.get_variable(
                    name='bias',
                    shape=(lfilter_shape[3]),
                    initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)

            self.push_to_terminal(x)

            return x


class TrConvLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params

    # lfilter_shape = H x W x Cout x Cin
    def op(self,
           lfilter_shape=(3, 3, 1, 1),
           istride=2,
           spadding='SAME',
           buse_bias=False,
           sname=None,
           *args,
           **kwargs):
        tinputs = self.retrieve_from_terminal()
        # output shape for 2x (H,W) by istride=2
        out_shape = tf.multiply(tf.shape(tinputs), (1, istride, istride, 1))
        # filter_shape (k,k,Cout,Cin)
        out_shape = tf.stack(
            (out_shape[0], out_shape[1], out_shape[2], lfilter_shape[2]))

        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            # netparam as default
            sinitializer = kwargs.get('sinitializer',
                                      self.net_params.initializer)
            # INITIALIZERS
            init_fn = CommonFunc.get_initializer(sinitializer)
            # Kernel
            kernels = tf.get_variable(name='kernel',
                                      shape=lfilter_shape,
                                      initializer=init_fn)
            # Padding
            if spadding == 'REFLECT':
                pad1 = tf.cast(tf.subtract(lfilter_shape[0:2], 1) / 2,
                               dtype=tf.int32)
                pad2 = tf.cast(tf.div(lfilter_shape[0:2], 2), dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0], pad2[0]], [pad1[1], pad2[1]],
                            [0, 0]]
                tinputs = tf.pad(tinputs, pad_size, 'REFLECT')
                spadding = 'VALID'
            # Transposed convolution
            x = tf.nn.conv2d_transpose(tinputs,
                                       kernels,
                                       out_shape,
                                       strides=istride,
                                       padding=spadding)
            if buse_bias:
                bias = tf.get_variable(
                    name='bias',
                    shape=[lfilter_shape[2]],
                    initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)

            self.push_to_terminal(x)

            return x


class MaxPoolLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params

    def op(self, ipool_size=2, istride=2, spadding='VALID', sname=None):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            if spadding == 'REFLECT':
                pad1 = tf.cast(tf.subtract((ipool_size, ipool_size), 1) / 2,
                               dtype=tf.int32)
                pad2 = tf.cast(tf.cast(
                    (ipool_size, ipool_size), dtype=tf.float32) / 2.,
                               dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0], pad2[0]], [pad1[1], pad2[1]],
                            [0, 0]]
                tinputs = tf.pad(tinputs, pad_size, 'REFLECT')

                spadding = 'VALID'

            x = tf.nn.max_pool2d(tinputs,
                                 ipool_size,
                                 istride,
                                 spadding,
                                 name=sname + '_Maxpool')
            self.push_to_terminal(x)

            return x


class AvgPoolLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params

    def op(self, ipool_size=2, istride=2, spadding='VALID', sname=None):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            if spadding == 'REFLECT':
                pad1 = tf.cast(tf.subtract((ipool_size, ipool_size), 1) / 2,
                               dtype=tf.int32)
                pad2 = tf.cast(tf.cast(
                    (ipool_size, ipool_size), dtype=tf.float32) / 2.,
                               dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0], pad2[0]], [pad1[1], pad2[1]],
                            [0, 0]]
                tinputs = tf.pad(tinputs, pad_size, 'REFLECT')

                spadding = 'VALID'

            x = tf.nn.avg_pool2d(tinputs,
                                 ipool_size,
                                 istride,
                                 spadding,
                                 name=sname + '_Avgpool')
            self.push_to_terminal(x)

            return x


class AddOnLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params

    def activation(self, sactivation=None, sname=None):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            # net param as default
            if sactivation is None:
                sactivation = self.net_params.activation

            if sactivation == 'ReLu':
                x = tf.nn.relu(tinputs, name=sname + '_ReLu')
            elif sactivation == 'LReLu':
                x = tf.nn.leaky_relu(tinputs, name=sname + '_LReLu')
            elif sactivation == 'PReLu':
                slope = tf.get_variable(name='slope',
                                        initializer=tf.constant(0.2))
                x = tf.where(tf.less(tinputs, 0.),
                             tinputs * slope,
                             tinputs,
                             name=sname + '_PReLu')
            elif sactivation == 'TanH':
                x = tf.nn.tanh(tinputs, name=sname + '_TanH')
            elif sactivation == 'SWISH':
                x = tf.nn.swish(tinputs, name=sname + '_SWISH')
            else:
                raise NotImplementedError(
                    'sactivation parameter ({:s}) is not defined'.format(
                        sactivation))
            self.push_to_terminal(x)

            return x

    def flatten(self, sname=None):
        tinputs = self.retrieve_from_terminal()
        x = tf.reshape(tinputs, (tf.shape(tinputs)[0], -1))
        self.push_to_terminal(x)

        return x

    def dropout(self, frate=None, sname=None):
        if self.net_params.use_drop is False:
            return

        if frate is None:
            frate = self.net_params.drop_rate
        tinputs = self.retrieve_from_terminal()
        x = tf.cond(self.net_params.ph_use_drop,
                    lambda: tf.nn.dropout(tinputs, rate=frate),
                    lambda: tinputs)
        self.push_to_terminal(x)

        return x

    def batch_norm(self, iCin=10, smode='CONV', sname=None):
        if self.net_params.use_batch_norm is False:
            return None
        """
        CONV Mode - mean and variance for axis=[0, 1, 2] (NxHxWxC),
                    i.e., only channel dimension survives
        DENSE Mode - mean and variance for axis=[0] (NxC),
                    i.e., only batch dimension is normalized
        """
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            beta = tf.get_variable(
                name='beta',
                shape=(iCin),
                initializer=tf.constant_initializer(value=0.0))
            gamma = tf.get_variable(
                name='gamma',
                shape=(iCin),
                initializer=tf.constant_initializer(value=1.0))

            count = tf.get_variable(name='count',
                                    trainable=False,
                                    initializer=tf.constant(0.0))
            sh_x = tf.get_variable(
                name='sh_x',
                shape=(iCin),
                trainable=False,
                initializer=tf.constant_initializer(value=0.0))
            sh_x2 = tf.get_variable(
                name='sh_x2',
                shape=(iCin),
                trainable=False,
                initializer=tf.constant_initializer(value=0.0))

            # net_param: reset and train
            brst = self.net_params.ph_bn_reset
            btrain = self.net_params.ph_bn_train

            # mean, var for a mini-batch
            if smode == 'CONV':
                batch_mean, batch_var = tf.nn.moments(tinputs, [0, 1, 2],
                                                      name='moments')
            elif smode == 'DENSE':
                batch_mean, batch_var = tf.nn.moments(tinputs, [0],
                                                      name='moments')
            else:
                raise NotImplementedError(
                    '%s - mode of BN is not defined in %s' % (smode, sname))

            # mean(sh_x/count), var(sh_x2/count) for an entire batch
            # (for example, if reset_bn every other epoch, for two batches)
            # the followings are pseudo code:
            # if brst:
            # count.assign(1.0)
            # sh_x.assign(batch_mean)
            # sh_x2.assign(batch_var + batch_mean ** 2)
            # elif btrain:
            # count.assign_add(1.0)
            # sh_x.assign_add(batch_mean)
            # sh_x2.assign_add(batch_var + batch_mean ** 2)

            count, sh_x, sh_x2 = tf.cond(
                brst, lambda: (count.assign(1.0), sh_x.assign(batch_mean),
                               sh_x2.assign(batch_var + batch_mean**2)),
                lambda: (tf.cond(
                    btrain, lambda:
                    (count.assign_add(1.0), sh_x.assign_add(batch_mean),
                     sh_x2.assign_add(batch_var + batch_mean**2)), lambda:
                    (tf.identity(count), tf.identity(sh_x), tf.identity(sh_x2)
                     ))))

            def mean_var_with_update():
                # tf.control_dep determines the order of variables evalution
                # batch_mean, batch_var are depedent on count, sh_x, sh_x2,
                # therefore count, sh_x, sh_x2 should be evaluated
                # ahead of batch_mean and batch_var
                with tf.control_dependencies([count, sh_x, sh_x2]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # if it is in the middelf of train, batch_mean, batch_var of a minibatch
            # else, mean and var of an entire batch
            mean, var = tf.cond(
                btrain, mean_var_with_update, lambda:
                (sh_x / count, sh_x2 / count - (sh_x / count)**2))
            normed = tf.nn.batch_normalization(tinputs, mean, var, beta, gamma,
                                               1e-3)
            self.push_to_terminal(normed)

            return normed

    def ins_norm(self, iCin=10, sname=None):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            beta = tf.get_variable(
                name='beta',
                shape=(iCin),
                initializer=tf.constant_initializer(value=0.0))
            gamma = tf.get_variable(
                name='gamma',
                shape=(iCin),
                initializer=tf.constant_initializer(value=1.0))
            mu, sigma_sq = tf.nn.moments(
                tinputs, [1, 2],
                keep_dims=True)  # mean and variance of tinputs over H, W axis
            epsilon = 1e-3
            normed = (tinputs - mu) / ((sigma_sq + epsilon)**(.5)
                                       )  # normalize (mean 0 and variance 1)
            normed = gamma * normed + beta  # weight and bias over the normalized input
            self.push_to_terminal(normed)

            return normed

    def add(self, tbranch, sname=None):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            x = tf.math.add(tinputs, tbranch)
            self.push_to_terminal(x)

            return x

    def concat(self, lbranch, sname=None):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            x = self.retrieve_from_terminal()
            t_concat = [x, *lbranch]
            x = tf.concat(t_concat, axis=3)
            # push to the main terminal
            self.push_to_terminal(x)

            return x


# composites layers
class DenseActLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params
        self.dense_layer = DenseLayer(self.net_params, term_name=term_name)
        self.addon_layer = AddOnLayer(self.net_params, term_name=term_name)

    def op(self, sactivation=None, sname=None, *args, **kwargs):
        self.dense_layer.op(*args, **kwargs, sname=sname)
        # net param as default
        if sactivation is None:
            sactivation = self.net_params.activation
        self.addon_layer.activation(sactivation, sname)

        return self.retrieve_from_terminal()


class DenseBnActLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params
        self.dense_layer = DenseLayer(self.net_params, term_name=term_name)
        self.addon_layer = AddOnLayer(self.net_params, term_name=term_name)

    def op(self, iout_nodes=10, sactivation=None, sname=None, *args, **kwargs):
        self.dense_layer.op(iout_nodes=iout_nodes,
                            sname=sname,
                            *args,
                            **kwargs)
        self.addon_layer.batch_norm(iCin=iout_nodes,
                                    smode='DENSE',
                                    sname=sname + '_bn')
        # net param as default
        if sactivation is None:
            sactivation = self.net_params.activation
        self.addon_layer.activation(sactivation, sname)

        return self.retrieve_from_terminal()


class ConvActLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params
        self.conv_layer = ConvLayer(self.net_params, term_name=term_name)
        self.addon_layer = AddOnLayer(self.net_params, term_name=term_name)

    def op(self, sactivation=None, sname=None, *args, **kwargs):
        self.conv_layer.op(*args, **kwargs, sname=sname)
        # net param as default
        if sactivation is None:
            sactivation = self.net_params.activation
        self.addon_layer.activation(sactivation, sname)

        return self.retrieve_from_terminal()


# class TrConvActLayer(Layer):
# def __init__(self, net_params, term_name='main'):
# super().__init__(term_name)
# self.net_params = net_params
# self.trconv_layer = TrConvLayer(self.net_params, term_name=term_name)
# self.addon_layer = AddOnLayer(self.net_params, term_name=term_name)

# def op(self, sactivation=None, sname=None, *args, **kwargs):
# with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
# self.trconv_layer.op(*args, **kwargs, sname=sname)
# # net param as default
# if sactivation is None:
# sactivation = self.net_params.activation
# self.addon_layer.activation(sactivation, sname)

# return self.retrieve_from_terminal()


class ConvBnActLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params
        self.conv_layer = ConvLayer(self.net_params, term_name=term_name)
        self.addon_layer = AddOnLayer(self.net_params, term_name=term_name)

    def op(self,
           lfilter_shape=[3, 3, 1, 1],
           sactivation=None,
           sname=None,
           *args,
           **kwargs):
        self.conv_layer.op(lfilter_shape=lfilter_shape,
                           sname=sname,
                           *args,
                           **kwargs)
        self.addon_layer.batch_norm(iCin=lfilter_shape[3],
                                    smode='CONV',
                                    sname=sname + '_bn')
        # net param as default
        if sactivation is None:
            sactivation = self.net_params.activation
        self.addon_layer.activation(sactivation, sname)

        return self.retrieve_from_terminal()


class ConvInActLayer(Layer):
    def __init__(self, net_params, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params
        self.conv_layer = ConvLayer(self.net_params, term_name=term_name)
        self.addon_layer = AddOnLayer(self.net_params, term_name=term_name)

    def op(self,
           lfilter_shape=[3, 3, 1, 1],
           sactivation=None,
           sname=None,
           *args,
           **kwargs):
        self.conv_layer.op(lfilter_shape=lfilter_shape,
                           sname=sname,
                           *args,
                           **kwargs)
        self.addon_layer.ins_norm(iCin=lfilter_shape[3], sname=sname + '_in')
        # net param as default
        if sactivation is None:
            sactivation = self.net_params.activation
        self.addon_layer.activation(sactivation, sname)

        return self.retrieve_from_terminal()


''' Multi-terminal layers '''


class ResBlockLayer(Layer):
    def __init__(self, net_params, term_name='main', type='SHORT'):
        super().__init__(term_name)
        self.net_params = net_params
        self.type = type
        if self.net_params.use_batch_norm:
            self.conv_layer = ConvBnActLayer(self.net_params)
        else:
            self.conv_layer = ConvActLayer(self.net_params)
        self.conv2_layer = ConvLayer(self.net_params)
        self.avgpool = AvgPoolLayer(self.net_params)

    def op(self, iCin, iCout, istride, sname=None):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            tinputs = self.retrieve_from_terminal()
            # type='SHORT' - 3x3-3x3, 'LONG' - 1x1-3x3-1x1
            ###################################################
            # we use branch1 terminal in order to use the layer function;
            # avgpool at the initial stage of resblock
            # For identity mapping, the dimension of the input and the output should be matched
            # HxW: if the initial stride convolution should be accounted by avgpool of the input
            self.avgpool.create_terminal(
                term_name='brach1').push_to_terminal(tinputs)
            if istride != 1:
                self.avgpool.op(spadding='SAME', sname=sname + '0')
                branch1_out = self.avgpool.retrieve_from_terminal()
            else:
                branch1_out = tinputs
            # C: if the input and ouput channels are different, padding is needed
            # (C of the input < C of the output)
            if iCin != iCout:
                # C tpadding
                c_pad1 = (iCout - iCin) // 2
                c_pad2 = (iCout - iCin) - c_pad1
                c_padding = [c_pad1, c_pad2]
                tpadding = tf.constant([[0, 0], [0, 0], [0, 0], c_padding])
                branch1_out = tf.pad(branch1_out, tpadding, 'CONSTANT')

            if self.type == 'SHORT':
                self.conv_layer.op(lfilter_shape=(3, 3, iCin, iCout),
                                   istride=istride,
                                   sname=sname + '_conv1')
                self.conv2_layer.op(lfilter_shape=(3, 3, iCout, iCout),
                                    istride=1,
                                    sname=sname + '_conv2')
            elif self.type == 'LONG':
                self.conv_layer.op(lfilter_shape=(1, 1, iCin, iCout // 4),
                                   istride=istride,
                                   sname=sname + '_conv1')
                self.conv_layer.op(lfilter_shape=(3, 3, iCout // 4,
                                                  iCout // 4),
                                   istride=1,
                                   sname=sname + '+_conv2')
                self.conv2_layer.op(lfilter_shape=(1, 1, iCout // 4, iCout),
                                    istride=1,
                                    sname=sname + '_conv3')

            # identity mapping
            x = tf.math.add(branch1_out, self.retrieve_from_terminal())
            self.push_to_terminal(x)

            return x


class ConvBlockLayer(Layer):
    def __init__(self, net_params, binst_norm=False, term_name='main'):
        super().__init__(term_name)
        self.net_params = net_params
        if binst_norm:
            self.convact_layer = ConvInActLayer(self.net_params,
                                                term_name=term_name)
        else:
            self.convact_layer = ConvActLayer(self.net_params,
                                              term_name=term_name)
        self.conv_sizes = [3, 5, 7, 9]

    def op(self, max_size=3, ch_maps=(1, 1), buse_bias=False, sname=None):
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            # main terminal
            tinputs = self.retrieve_from_terminal()
            output_tensor = None

            for conv_size in [x for x in self.conv_sizes if x <= max_size]:
                sname_sub = f'{sname}_{conv_size}'
                self.push_to_terminal(tinputs)
                self.convact_layer.op(lfilter_shape=(conv_size, conv_size,
                                                     ch_maps[0], ch_maps[1]),
                                      istride=1,
                                      sname=sname_sub + '_1')
                self.convact_layer.op(lfilter_shape=(conv_size, conv_size,
                                                     ch_maps[1], ch_maps[1]),
                                      istride=1,
                                      sname=sname_sub + '_2')
                concat_tensor = self.convact_layer.retrieve_from_terminal()

                if output_tensor is not None:
                    output_tensor = tf.concat((output_tensor, concat_tensor),
                                              axis=3)
                else:
                    output_tensor = concat_tensor
            # push to the main terminal
            self.push_to_terminal(output_tensor)

            return output_tensor
