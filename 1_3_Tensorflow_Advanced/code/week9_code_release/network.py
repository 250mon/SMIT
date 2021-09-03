import tensorflow.compat.v1 as tf


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
    
        
    def initializer(self, sinitializer='he_normal'):
        
        if sinitializer == 'glorot_normal':
            init_fn = tf.initializers.glorot_normal()
        elif sinitializer == 'glorot_uniform':
            init_fn = tf.initializers.glorot_uniform()
        elif sinitializer == 'he_normal':
            init_fn = tf.initializers.he_normal()
        elif sinitializer == 'he_uniform':
            init_fn = tf.initializers.he_uniform()
        elif sinitializer == 'lecun_uniform':
            init_fn = tf.initializers.lecun_uniform()
        elif sinitializer == 'lecun_normal':
            init_fn = tf.initializers.lecun_normal()
        else:
            raise NotImplementedError('sinitializer parameter ({:s}) is not defined'.format(sinitializer))
            
        return init_fn
    
    @layer    
    def _batch_norm(self, tinputs, iCin=10, smode='CONV', btrain=True, sname=None):
        
        """
        CONV Mode - mean and variance for axis=[0, 1, 2] (NxHxWxC), i.e., only channel dimension is survived
        DENSE Mode - mean and variance for axis=[0] (NxC), i.e., only batch dimension is normalized
        """
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
                        
            beta = tf.get_variable(name='beta', shape=(iCin), initializer=tf.constant_initializer(value=0.0))
            gamma = tf.get_variable(name='gamma', shape=(iCin), initializer=tf.constant_initializer(value=1.0))
            
            #beta = tf.get_variable(name='beta', shape=(1), initializer=tf.constant_initializer(value=0.0))
            #gamma = tf.get_variable(name='gamma', shape=(1), initializer=tf.constant_initializer(value=1.0))
            
            if smode == 'CONV':
                               
                batch_mean, batch_var = tf.nn.moments(tinputs, [0,1,2], name='moments')
            
            elif smode == 'DENSE':
            
                batch_mean, batch_var = tf.nn.moments(tinputs, [0], name='moments')
            
            else:
                raise NotImplementedError('%s - mode of BN is not defined in %s'%(smode, sname))
    
            ema = tf.train.ExponentialMovingAverage(decay=0.99)
    
            def mean_var_with_update():
    
                ema_apply_op = ema.apply([batch_mean, batch_var])
    
                with tf.control_dependencies([ema_apply_op]):
    
                    return tf.identity(batch_mean), tf.identity(batch_var)
                                        
            mean, var = tf.cond(btrain, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
            #mean, var = tf.cond(btrain, mean_var_with_update, lambda: (tf.identity(batch_mean), tf.identity(batch_var)))
                        
            normed = tf.nn.batch_normalization(tinputs, mean, var, beta, gamma, 1e-3)
                
        return normed
    
    @layer
    def batch_norm(self, tinputs, iCin=10, brst=False, smode='CONV', btrain=True, sname=None):
        """
        CONV Mode - mean and variance for axis=[0, 1, 2] (NxHxWxC), i.e., only channel dimension is survived
        DENSE Mode - mean and variance for axis=[0] (NxC), i.e., only batch dimension is normalized
        """
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
                        
            beta = tf.get_variable(name='beta', shape=(iCin), initializer=tf.constant_initializer(value=0.0))
            gamma = tf.get_variable(name='gamma', shape=(iCin), initializer=tf.constant_initializer(value=1.0))
            
            
            # SJY:
            # the whole idea of making count, x, x2 be tf.variables is due to keep the states of these variables during the session
            # if these are not trainable, why not using a python class to keep these states
            count = tf.get_variable(name='count', trainable=False, initializer=tf.constant(0.0))
            sh_x = tf.get_variable(name='sh_x', shape=(iCin), trainable=False, initializer=tf.constant_initializer(value=0.0))
            sh_x2 = tf.get_variable(name='sh_x2', shape=(iCin), trainable=False, initializer=tf.constant_initializer(value=0.0))
                                    
            # mean, var for a mini-batch
            if smode == 'CONV':
                batch_mean, batch_var = tf.nn.moments(tinputs, [0,1,2], name='moments')
            elif smode == 'DENSE':
                batch_mean, batch_var = tf.nn.moments(tinputs, [0], name='moments')
            else:
                raise NotImplementedError('%s - mode of BN is not defined in %s'%(smode, sname))
            # mean(sh_x/count), var(sh_x2/count) for an entire batch (for example, if reset every other epoch, for two batches)
            # if reset:
            #   count=1; sh_x=mean, sh_x2=(var+mean**2)
            # else:
            #   if train: 
            #       count=count+1, sh_x+=batch_mean, sh_x2+=var+mean**2
            #   else:     
            #       count, sh_x, sh_x2
            count, sh_x, sh_x2 = tf.cond(brst, 
                                         lambda:(count.assign(1.0), sh_x.assign(batch_mean), sh_x2.assign(batch_var+batch_mean**2)), 
                                         lambda:(tf.cond(btrain, 
                                                         lambda:(count.assign_add(1.0), sh_x.assign_add(batch_mean), sh_x2.assign_add(batch_var+batch_mean**2)), 
                                                         lambda:(tf.identity(count), tf.identity(sh_x), tf.identity(sh_x2)))))

            def mean_var_with_update():
                # tf.control_dep determines the order of evalution the variables
                # batch_mean, batch_var are depedent on count, sh_x, sh_x2, 
                # therefore count, sh_x, sh_x2 should be evaluated ahead of batch_mean and batch_var
                with tf.control_dependencies([count, sh_x, sh_x2]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            # if train, batch_mean, batch_var of a minibatch
            # else, mean and var of an entire batch
            mean, var = tf.cond(btrain, mean_var_with_update, lambda: (sh_x/count, sh_x2/count-(sh_x/count)**2))
            normed = tf.nn.batch_normalization(tinputs, mean, var, beta, gamma, 1e-3)
        return normed
                        
    
    @layer
    def dense(self, tinputs, iin_nodes=10, iout_nodes=10, buse_bias=True, sinitializer='he_uniform', sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            # INITIALIZERS
            init_fn = self.initializer(sinitializer)
            
            # WEIGHT VARIABLE    
            weights = tf.get_variable(name='weights', shape=(iin_nodes, iout_nodes), initializer=init_fn)
            x = tf.matmul(tinputs, weights) # Nxin_nodes multiply in_nodesxnum_nodes
            # BIAS
            if buse_bias:
                bias = tf.get_variable(name='bias', shape=(iout_nodes), initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)
                                                        
            return x
    
    
    @layer
    def denseact(self, tinputs, iin_nodes=10, iout_nodes=10, buse_bias=True, sinitializer='he_uniform', sactivation='ReLu', sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            (self.feed(tinputs)
                 .dense(iin_nodes=iin_nodes, iout_nodes=iout_nodes, buse_bias=buse_bias, sinitializer=sinitializer, sname=sname)
                 .activation(sactivation=sactivation, sname=sname))
            
            return self.terminals[0]
        
        
    @layer
    def densebnact(self, tinputs, iin_nodes=10, iout_nodes=10, buse_bias=True, sinitializer='he_uniform', sactivation='ReLu', brst=False, btrain=True, sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            (self.feed(tinputs)
                 .dense(iin_nodes=iin_nodes, iout_nodes=iout_nodes, buse_bias=buse_bias, sinitializer=sinitializer, sname=sname)
                 .batch_norm(iCin=iout_nodes, brst=brst, smode='DENSE', btrain=btrain, sname=sname+'_bn')
                 .activation(sactivation=sactivation, sname=sname))
            
            return self.terminals[0]
        
    
    @layer
    def conv(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            # INITIALIZERS
            init_fn = self.initializer(sinitializer)
                
            kernels = tf.get_variable(name='kernel', shape=lfilter_shape, initializer=init_fn)
            
            if spadding == 'REFLECT':
                
                pad1 = tf.cast(tf.subtract(lfilter_shape[0:2], 1)/2, dtype=tf.int32)
                pad2 = tf.cast(tf.cast(lfilter_shape[0:2], dtype=tf.float32)/2., dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
                tinputs = tf.pad(tinputs, pad_size, 'REFLECT')
                
                spadding = 'VALID'
            
            x = tf.nn.conv2d(tinputs, kernels, strides=(1, istride, istride, 1), padding=spadding)
            
            if buse_bias:
                bias = tf.get_variable(name='bias', shape=(lfilter_shape[3]), initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)
                            
            return x
    
    
    @layer    
    def activation(self, tinputs, sactivation='ReLu', sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            if sactivation == 'ReLu':
                x = tf.nn.relu(tinputs)
            elif sactivation == 'LReLu':
                x = tf.nn.leaky_relu(tinputs)
            elif sactivation == 'PReLu':
                slope = tf.get_variable(name='slope', initializer=tf.constant(0.2))
                x = tf.where(tf.less(tinputs, 0.), tinputs*slope, tinputs)
            elif sactivation == 'SWISH':
                x = tf.nn.swish(tinputs)
            else:
                raise NotImplementedError('sactivation parameter ({:s}) is not defined'.format(sactivation))
                
            return x


    @layer
    def convact(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sactivation='ReLu', sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            (self.feed(tinputs)
                 .conv(lfilter_shape=lfilter_shape, istride=istride, spadding=spadding, buse_bias=buse_bias, sinitializer=sinitializer, sname=sname)
                 .activation(sactivation=sactivation, sname=sname))
                            
            return self.terminals[0]
    
    
    @layer
    def convbnact(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sactivation='ReLu', brst=False, btrain=True, sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            (self.feed(tinputs)
                 .conv(lfilter_shape=lfilter_shape, istride=istride, spadding=spadding, buse_bias=buse_bias, sinitializer=sinitializer, sname=sname)
                 .batch_norm(iCin=lfilter_shape[3], brst=brst, smode='CONV', btrain=btrain, sname=sname+'_bn')
                 .activation(sactivation=sactivation, sname=sname))
                            
            return self.terminals[0]
            
        
    @layer
    def maxpool(self, tinputs, ipool_size=2, istride=2, spadding='VALID', sname=None):
        
        if spadding == 'REFLECT':
            
            pad1 = tf.cast(tf.subtract((ipool_size, ipool_size), 1)/2, dtype=tf.int32)
            pad2 = tf.cast(tf.cast((ipool_size, ipool_size), dtype=tf.float32)/2., dtype=tf.int32)
            pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
            tinputs = tf.pad(tinputs, pad_size, 'REFLECT')
            
            spadding = 'VALID'
        
        x = tf.nn.max_pool2d(tinputs, ipool_size, istride, spadding)
                                    
        return x
        
    @layer
    def avgpool(self, tinputs, ipool_size=2, istride=2, spadding='VALID', sname=None):
        
        if spadding == 'REFLECT':
            
            pad1 = tf.cast(tf.subtract((ipool_size, ipool_size), 1)/2, dtype=tf.int32)
            pad2 = tf.cast(tf.cast((ipool_size, ipool_size), dtype=tf.float32)/2., dtype=tf.int32)
            pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
            tinputs = tf.pad(tinputs, pad_size, 'REFLECT')
            
            spadding = 'VALID'
        
        x = tf.nn.avg_pool2d(tinputs, ipool_size, istride, spadding)
                                    
        return x
        
    
    @layer
    def dropout(self, tinputs, frate=0.2, buse_drop=True, sname=None):
        
        x = tf.cond(buse_drop, lambda: tf.nn.dropout(tinputs, rate=frate), lambda: tinputs)
        
        return x
        
    
    @layer
    def flatten(self, tinputs, sname=None):
                                                            
            return tf.reshape(tinputs, (tf.shape(tinputs)[0], -1))
        
        
    @layer
    def resblk(self, tinputs, iCin, iCout, istride=1, buse_bias=False, sinit='he_normal', stype='SHORT', sactivation='ReLu', sname=None):
        
        padding='REFLECT'
                
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            if istride != 1:
                side = tf.nn.avg_pool2d(tinputs, (istride, istride), (istride, istride), 'SAME')
            else:
                side = tinputs
                #side = tf.identity(tinputs)
                
            if iCin != iCout:
                pad1 = (iCout-iCin)//2
                c_pad = [pad1, (iCout-iCin)-pad1]
                side = tf.pad(side, [[0,0], [0,0], [0,0], c_pad])
                        
            if stype=='SHORT':
                (self.feed(tinputs)
                     .convact(lfilter_shape=(3,3,iCin,iCout), istride=istride, spadding=padding, buse_bias=buse_bias, sinitializer=sinit, sactivation=sactivation, sname='conv1')
                     .conv(lfilter_shape=(3,3,iCout,iCout), spadding=padding, buse_bias=buse_bias, sinitializer=sinit, sname='conv2'))
            elif stype=='LONG':
                Cmid = iCout//4
                (self.feed(tinputs)
                     .convact(lfilter_shape=(1,1,iCin,Cmid), spadding=padding, buse_bias=buse_bias, sactivation=sactivation,  sinitializer=sinit, sname='conv1')
                     .convact(lfilter_shape=(3,3,Cmid,Cmid), istride=istride, spadding=padding, buse_bias=buse_bias, sactivation=sactivation,  sinitializer=sinit, sname='conv2')
                     .conv(lfilter_shape=(1,1,Cmid,iCout), spadding=padding, buse_bias=buse_bias, sinitializer=sinit, sname='conv3'))
            else:
                raise NotImplementedError('ctype of resblock is not defined in %s'%sname)
            
            x = side + self.terminals[0]
            
                
            return x
        
        
        
    
    
