import tensorflow as tf

DEFAULT_ACTIVATION = tf.nn.leaky_relu
DEFAULT_INITIALIZER = tf.initializers.glorot_normal()


def layer(op):
    '''Decorator for chaining components of layer'''
    def layer_decorated(self, *args, **kwargs):
        
        name = kwargs.setdefault('sname', 'no_given_name')
        
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            raise NotImplementedError('List Inputs - Not implemented yet %s.' %name)
        
        layer_output = op(self, layer_input, *args, **kwargs)
        
        self.feed(layer_output)
        
        return self

    return layer_decorated



class Network(object):
    def __init__(self):

        # network terminal node
        self.terminals = []
        self._build()

    def _build(self):
        '''Construct network model. '''
        raise NotImplementedError('Must be implemented by the subclass in model.py')
        
    def feed(self, tensor):
        
        self.terminals = []
        self.terminals.append(tensor)
            
        return self
    
    @layer
    def dense(self, tinputs, iin_nodes=10, iout_nodes=10, buse_bias=True, sactivation='ReLu', sinitializer='he_uniform', sname=None):
        
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
            x = tf.matmul(tinputs, weights) # Nxin_nodes multiply in_nodesxnum_nodes
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
        
    
    @layer
    def conv(self, tinputs, lfilter_shape=(3,3,1,1), lstrides=(1,1,1,1), spadding='SAME', buse_bias=False, sactivation=None,  sinitializer='he_normal', sname=None):
        
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
                
            kernels = tf.get_variable(name='kernel', shape=lfilter_shape, initializer=init_fn)
            
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

