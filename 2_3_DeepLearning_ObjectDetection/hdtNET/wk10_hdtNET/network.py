# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:00:25 2019

@author: Angelo
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

DEFAULT_ACTIVATION = tf.nn.relu
DEFAULT_INITIALIZER = tf.initializers.glorot_normal()

def layer(op):
    '''Decorator for chaining components of layer'''
    def layer_decorated(self, *args, **kwargs):
        
        name = kwargs.setdefault('name', 'no_given_name')
        
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        
        layer_output = op(self, layer_input, *args, **kwargs)
        
        self.feed(layer_output)
        
        return self

    return layer_decorated



class Network(object):
    def __init__(self):

        # network terminal node
        self.terminals = []
        self._build()

    def _build(self, is_training):
        '''Construct network model. '''
        raise NotImplementedError('Must be implemented by the subclass.')
        
    def feed(self, tensor):
        
        self.terminals = []
        self.terminals.append(tensor)
            
        return self
    
    @layer
    def conv(self, inputs, filters, kernel_size=3, rate=1, strides=1, padding='SAME',
             activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER, name=None):
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            output = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, dilation_rate=rate,
                                      strides=strides, padding=padding, activation=activation, use_bias=use_bias, 
                                      kernel_initializer=kernel_initializer, name=name)
                
        return output
    
    @layer
    def deconv(self, x, out_channel, kernel=4, stride=2, use_bias=True, name=None):
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
                        
            x = tf.layers.conv2d_transpose(inputs=x, filters=out_channel,
                                       kernel_size=kernel, kernel_initializer=DEFAULT_INITIALIZER,
                                       strides=stride, padding='SAME', use_bias=use_bias)

        return x
    
    @layer
    def deconv_nn(self, inputs, filters, rate=1, strides=(1,2,2,1), padding='SAME',
             activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER, name=None):
    
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            #ns = tf.contrib.framework.get_name_scope()
            kernels = tf.get_variable(name='kernel', shape=filters, initializer=kernel_initializer)
            num_pad = 0
            
            if padding == 'REFLECT':
                
                num_pad = tf.cast(tf.div(filters[0], 2), dtype=tf.int32)
                pad_size = [[0, 0], [num_pad,num_pad], [num_pad, num_pad], [0,0]]
                inputs = tf.pad(inputs, pad_size, 'REFLECT')
                
                padding = 'SAME'
                
            out_shape = tf.multiply(tf.shape(inputs), strides)
            out_shape = tf.stack((out_shape[0], out_shape[1], out_shape[2], filters[2]))
                
            x = tf.nn.conv2d_transpose(inputs, kernels, out_shape, strides=strides, padding=padding)
            
            if use_bias:
                bias = tf.get_variable(name='bias', shape=[filters[2]], initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)
                
            if activation is not None:
                x = activation(x)
                
            x = tf.cond(tf.equal(num_pad, 0), lambda: x, lambda: x[:, 2*num_pad:-2*num_pad, 2*num_pad:-2*num_pad, :])
                            
            return x
    
    
    @layer
    def conv_nn(self, inputs, filters, rate=1, strides=[1,1,1,1], padding='SAME',
             activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER, name=None):
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            #ns = tf.contrib.framework.get_name_scope()
            kernels = tf.get_variable(name='kernel', shape=filters, initializer=kernel_initializer)
            
            if padding == 'REFLECT':
                
                pad1 = tf.cast(tf.subtract(filters[0:2], 1)/2, dtype=tf.int32)
                pad2 = tf.cast(tf.div(filters[0:2], 2), dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
                inputs = tf.pad(inputs, pad_size, 'REFLECT')
                
                padding = 'VALID'
            
            x = tf.nn.conv2d(inputs, kernels, dilations=[1,rate,rate,1], strides=strides, padding=padding)
            
            if use_bias:
                bias = tf.get_variable(name='bias', shape=[filters[3]], initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)
                
            if activation is not None:
                x = activation(x)
                
            return x
        
    @layer
    def conv_dw(self, inputs, filters, strides=[1,1,1,1], padding='SAME', rate=(1,1),
             activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER, name=None):
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            #filters of dimension (kxkxC-inxC_mul) - output channels of C_inxC_mul
            kernels = tf.get_variable(name='kernel', shape=filters, initializer=kernel_initializer)
            
            if padding == 'REFLECT':
                
                pad1 = tf.cast(tf.subtract(filters[0:2], 1)/2, dtype=tf.int32)
                pad2 = tf.cast(tf.div(filters[0:2], 2), dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
                inputs = tf.pad(inputs, pad_size, 'REFLECT')
                
                padding = 'VALID'
            
            x = tf.nn.depthwise_conv2d(inputs, kernels, strides=strides, padding=padding, rate=rate)
            
            if use_bias:
                bias = tf.get_variable(name='bias', shape=[filters[3]], initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)
                
            if activation is not None:
                x = activation(x)
                
            return x
        
        
    @layer
    def batch_normalization(self, inputs, name=None, training=True, activation=DEFAULT_ACTIVATION):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            output = tf.layers.batch_normalization(inputs, momentum=0.95, epsilon=1e-5, training=training)
        
            '''
            # NOTE: when training, the moving_mean and moving_variance need to be updated. 
            By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, 
            so they need to be executed alongside the train_op, i.e., 
            
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = optimizer.minimize(loss)
            train_op = tf.group([train_op, update_ops])
            '''
        
            if activation is not None:
                output = activation(output)

            return output
        
    @layer
    def instance_normalization(self, inputs, name=None, training=True, activation=DEFAULT_ACTIVATION):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            output = tf.contrib.layers.instance_norm(inputs, activation_fn=activation)

            return output

    @layer
    def activator(self, inputs, name=None):
        return DEFAULT_ACTIVATION(inputs, name=name)
    
    
    @layer
    def max_pool(self, inputs, pool_size=2, strides=2, padding='SAME', name=None):
        return tf.layers.max_pooling2d(inputs, pool_size, strides,
                                       padding=padding, name=name)
    
    @layer
    def dense(self, inputs, units=1000, activation=None, use_bias=True, 
              kernel_initializer=DEFAULT_INITIALIZER, name=None):
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(inputs=inputs, units=units, activation=activation,
                                     use_bias=use_bias, kernel_initializer=kernel_initializer)
            
            return output
    
    
    @layer
    def avg_pool(self, inputs, pool_size=2, strides=2, padding='VALID', name=None):
        return tf.layers.average_pooling2d(inputs=inputs, pool_size=pool_size, strides=strides, name=name)
    
    
    @layer
    def resize_bilinear(self, inputs, size, name):
        return tf.image.resize_bilinear(inputs, size=size, align_corners=True, name=name)
        
                    
    @layer
    def d_conv(self, inputs, filters, strides=1, rate=2, padding='SAME',
             activation=DEFAULT_ACTIVATION, use_bias=False, kernel_initializer=DEFAULT_INITIALIZER, name=None):
        
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            #ns = tf.contrib.framework.get_name_scope()
            kernel = tf.get_variable(name='kernel', shape=filters, initializer=kernel_initializer)
            
            x = tf.nn.atrous_conv2d(inputs, kernel, rate, padding)
            
            if use_bias:
                bias = tf.get_variable(name='bias', shape=[filters[3]], initializer=tf.constant_initializer(value=0))
                x = tf.nn.bias_add(x, bias)
                
            if activation is not None:
                x = activation(x)
                
            return x
        
    
    @layer
    def resize_nn(self, inputs, size, name):
        return tf.image.resize_nearest_neighbor(inputs, size, name=name)
    
