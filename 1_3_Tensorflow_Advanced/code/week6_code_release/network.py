import tensorflow.compat.v1 as tf
import pdb


def layer(op):
    '''Decorator for chaining components of layer'''
    def layer_decorated(self, *args, **kwargs):
        
        name = kwargs.setdefault('sname', 'no_given_name')
        sterminal_key = kwargs.setdefault('sterminal_key', 'main')
        
        if len(self.terminals[sterminal_key]) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals[sterminal_key]) == 1:
            layer_input = self.terminals[sterminal_key][0]
        else:
            raise NotImplementedError('List Inputs - Not implemented yet %s.' % name)
        
        # forward pass
        layer_output = op(self, layer_input, *args, **kwargs)
        # feed the output from the previous layer to the next layer as an input
        self.feed(layer_output, sterminal_key)
        
        return self

    return layer_decorated



class Network(object):
    def __init__(self):

        # network terminal node
        self.terminals = {'main': []}
        self._build()

    def _build(self):
        '''Construct network model. '''
        raise NotImplementedError('Must be implemented by the subclass in model.py')
        
    def feed(self, tensor, sterminal_key='main'):
        
        self.terminals.setdefault(sterminal_key, [])
        self.terminals[sterminal_key] = []
        self.terminals[sterminal_key].append(tensor)
            
        return self
    
    """
    Parameters of dense network
    * tinputs:        input tensor
    * iin_nodes:      number of input nodes
    * iout_nodes:     number of output nodes
    * buse_bias:      whether or not to use bias
    * sactivation:    activation function
    * sinitializer:   initializer function
    * sname:          layer name
    """
    @layer
    def dense(self, tinputs, iin_nodes=10, iout_nodes=10, buse_bias=True, sactivation='ReLu', sinitializer='he_uniform', sname=None, sterminal_key='main'):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            # INITIALIZERS
            if sinitializer == 'glorot_normal':
                init_fn = tf.initializers.glorot_normal()
            elif sinitializer == 'glorot_uniform':
                init_fn = tf.initializers.glorot_uniform()
            elif sinitializer == 'he_normal':
                init_fn = tf.initializers.he_normal()
            else:
                init_fn = tf.initializers.he_uniform()
            # WEIGHT VARIABLE    
            weights = tf.get_variable(name='weights', shape=(iin_nodes, iout_nodes), initializer=init_fn)
            x = tf.matmul(tinputs, weights) # (N x in_nodes) multiply (in_nodes x out_nodes) = (N x out_nodes)
            # BIAS
            if buse_bias:
                bias = tf.get_variable(name='bias', shape=(iout_nodes), initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)
            #ACTIVATION
            if sactivation != 'None':
                if sactivation == 'ReLu':
                    x = tf.nn.relu(x)
                elif sactivation == 'LReLu':
                    x = tf.nn.leaky_relu(x)
                elif sactivation == 'PReLu':
                    slope = tf.get_variable(name='slope', initializer=tf.constant(0.2))
                    x = tf.where(tf.less(x, 0.), x*slope, x)
                else:
                    raise NotImplementedError('sactivation parameter is not defined in %s'%sname)
                            
            return x

    """
    Parameters of convolution layer
    * tinputs:        input tensor (N x H x W x C); convolution done for the last 3 dim
    * filters:        shape of 4D tensor (filter_H x filter_W x in_C x out_C)
    * lstrides:       list of ints that has length 1, 2, or 4
    * padding:        'SAME' for zero padding or 'VALID' for no padding
    * buse_bias:      whether or not to use bias
    * sactivation:    activation function
    * sinitializer:   initializer function
    * sname:          layer name
    """
    @layer
    def conv(self, tinputs, lfilter_shape=(3,3,1,1), lstrides=(1,1,1,1), spadding='SAME', buse_bias=False, sactivation=None,  sinitializer='he_normal', sname=None, sterminal_key='main'):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            # INITIALIZERS
            if sinitializer == 'glorot_normal':
                init_fn = tf.initializers.glorot_normal()
            elif sinitializer == 'glorot_uniform':
                init_fn = tf.initializers.glorot_uniform()
            elif sinitializer == 'he_normal':
                init_fn = tf.initializers.he_normal()
            else:
                init_fn = tf.initializers.he_uniform()

            # Kernel
            kernels = tf.get_variable(name='kernel', shape=lfilter_shape, initializer=init_fn)
            
            # Padding
            if spadding == 'REFLECT':
                pad1 = tf.cast(tf.subtract(lfilter_shape[0:2], 1)/2, dtype=tf.int32)
                pad2 = tf.cast(tf.div(lfilter_shape[0:2], 2), dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
                tinputs = tf.pad(tinputs, pad_size, 'REFLECT')
                spadding = 'VALID'
            
            x = tf.nn.conv2d(tinputs, kernels, strides=lstrides, padding=spadding)
            
            if buse_bias:
                bias = tf.get_variable(name='bias', shape=(lfilter_shape[3]), initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)
                
            #ACTIVATION
            if sactivation != 'None':
                if sactivation == 'ReLu':
                    x = tf.nn.relu(x)
                elif sactivation == 'LReLu':
                    x = tf.nn.leaky_relu(x)
                elif sactivation == 'PReLu':
                    slope = tf.get_variable(name='slope', initializer=tf.constant(0.2))
                    x = tf.where(tf.less(x, 0.), x*slope, x)
                else:
                    raise NotImplementedError('sactivation parameter is not defined in %s'%sname)
                
            return x
        
    @layer
    def maxpool(self, tinputs, ipool_size=2, istrides=2, spadding='VALID', sname=None, sterminal_key='main'):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            if spadding == 'REFLECT':
                # The following padding scheme is for convolution
                #pad1 = tf.cast(tf.subtract((ipool_size, ipool_size), 1)/2, dtype=tf.int32)
                #pad2 = tf.cast(tf.cast((ipool_size, ipool_size), dtype=tf.float32)/2., dtype=tf.int32)
                #pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]

                # padding (N, H, W, C)
                # padding in height direction
                input_height = tf.shape(tinpus)[2]
                h_pad = ipool_size - (input_height % ipool_size)
                h_padding = [h_pad//2, (h_pad+1)//2]
                # padding in width direction
                input_width = tf.shape(tinpus)[2]
                w_pad = ipool_size - (input_width % ipool_size)
                w_padding = [w_pad//2, (w_pad+1)//2]
                
                tpadding = tf.constant([[0,0], h_padding, w_padding, [0,0]])
                tinputs = tf.pad(tinputs, tpadding, 'REFLECT')
                
                spadding = 'VALID'
            
            x = tf.nn.max_pool2d(tinputs, ipool_size, istrides, padding=spadding)
                                        
            return x
        
    @layer
    def avgpool(self, tinputs, ipool_size=2, istrides=2, spadding='VALID', sname=None, sterminal_key='main'):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            if spadding == 'REFLECT':
                # padding (N, H, W, C)
                # padding in height direction
                input_height = tf.shape(tinpus)[2]
                h_pad = ipool_size - (input_height % ipool_size)
                h_padding = [h_pad//2, (h_pad+1)//2]
                # padding in width direction
                input_width = tf.shape(tinpus)[2]
                w_pad = ipool_size - (input_width % ipool_size)
                w_padding = [w_pad//2, (w_pad+1)//2]
                
                tpadding = tf.constant([[0,0], h_padding, w_padding, [0,0]])
                tinputs = tf.pad(tinputs, tpadding, 'REFLECT')
                
                spadding = 'VALID'
            
            x = tf.nn.avg_pool2d(tinputs, ipool_size, istrides, padding=spadding)
                                        
            return x

    @layer
    def resblock(self, tinputs, iCin, iCout, lstrides, ctype='SHORT', sname=None, sterminal_key='main'):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            # DEFINE FUNCTION HERE 
            #ctype='SHORT' - 3x3-3x3, 'LONG' - 1x1-3x3-1x1
            ###################################################
            # we use branch1 terminal in order to use the layer function; avgpool at the initial stage of resblock
            self.feed(tinputs, sterminal_key='branch1')
            # For identity mapping, the dimension of the input and the output should be matched
            # HxW: if the initial stride convolution should be accounted by avgpool of the input
            if lstrides[1] != 1:
                # ToDo: it works only for lstrides 2
                self.avgpool(sname=sname+'0', sterminal_key='branch1')
            # C: if the input and ouput channels are different, padding is needed (C of the input < C of the output)
            if iCin != iCout:
                # C tpadding
                c_pad = iCout - iCin
                c_padding = [c_pad//2, c_pad//2]
                tpadding = tf.constant([[0,0], [0,0], [0,0], c_padding])
                tinputs = tf.pad(self.terminals['branch1'][0], tpadding, 'CONSTANT')

            if ctype is 'SHORT':
                (self.conv(lfilter_shape=(3,3,iCin,iCout), lstrides=lstrides, spadding='REFLECT', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname=sname+'conv1') 
                     .conv(lfilter_shape=(3,3,iCout,iCout), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname=sname+'conv2'))
            elif ctype is 'LONG':
                (self.conv(lfilter_shape=(1,1,iCin,iCout//4), lstrides=lstrides, spadding='REFLECT', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname=sname+'conv1') 
                     .conv(lfilter_shape=(3,3,iCout//4,iCout//4), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname=sname+'conv2')
                     .conv(lfilter_shape=(1,1,iCout//4,iCout), lstrides=(1,1,1,1), spadding='REFLECT', buse_bias=True, sactivation='ReLu', sinitializer='he_normal', sname=sname+'conv3'))

            # identity mapping
            x =  tf.math.add(tinputs, self.terminals['main'][0])
            return x

