import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

#from utils import ConvolutionOrthogonal2D, ConvolutionDeltaOrthogonal


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


class Initializer(object):
    """Initializer base class: all initializers inherit from this class."""
    
    def __call__(self, shape, dtype=None, partition_info=None):
        """Returns a tensor object initialized as specified by the initializer.
    
        Args:
          shape: Shape of the tensor.
          dtype: Optional dtype of the tensor. If not provided use the initializer
            dtype.
          partition_info: Optional information about the possible partitioning of a
            tensor.
        """
        raise NotImplementedError
        
    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
    
        Returns:
          A JSON-serializable Python dict.
        """
        return {}
    
    
    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.
    
        Example:
    
        ```python
        initializer = RandomUniform(-1, 1)
        config = initializer.get_config()
        initializer = RandomUniform.from_config(config)
        ```
    
        Args:
          config: A Python dictionary. It will typically be the output of
            `get_config`.
    
        Returns:
          An Initializer instance.
        """
        return cls(**config)

class ConvolutionDeltaOrthogonal(Initializer):
    """Initializer that generates a delta orthogonal kernel for ConvNets.

    The shape of the tensor must have length 3, 4 or 5. The number of input
    filters must not exceed the number of output filters. The center pixels of the
    tensor form an orthogonal matrix. Other pixels are set to be zero. See
    algorithm 2 in (Xiao et al., 2018).


    Args:
      gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
        The 2-norm of an input is multiplied by a factor of `gain` after applying
        this convolution.
      seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
      dtype: Default data type, used if no `dtype` argument is provided when
        calling the initializer. Only floating point types are supported.
    References:
        [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
        ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
    """

    def _assert_float_dtype(self, dtype):
        
        """
        Validate and return floating point type based on `dtype`.
        
        `dtype` must be a floating point type.
        
        Args:
            dtype: The data type to validate.
    
        Returns:
            Validated type.
    
        Raises:
            ValueError: if `dtype` is not a floating point type.
            
        """
  
        if not dtype.is_floating:
            raise ValueError("Expected floating point type, got %s." % dtype)
        return dtype

    def __init__(self, gain=1.0, seed=None, dtype=dtypes.float32):
      self.gain = gain
      self.dtype = self._assert_float_dtype(dtypes.as_dtype(dtype))
      self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
      if dtype is None:
          dtype = self.dtype
      # Check the shape
      if len(shape) < 3 or len(shape) > 5:
          raise ValueError("The tensor to initialize must be at least "
                           "three-dimensional and at most five-dimensional")

      if shape[-2] > shape[-1]:
          raise ValueError("In_filters cannot be greater than out_filters.")

      # Generate a random matrix
      a = random_ops.random_normal([shape[-1], shape[-1]],
                                   dtype=dtype,
                                   seed=self.seed)
      # Compute the qr factorization
      q, r = gen_linalg_ops.qr(a, full_matrices=False)
      # Make Q uniform
      d = array_ops.diag_part(r)
      q *= math_ops.sign(d)
      q = q[:shape[-2], :] #(cin, cout)
      q *= math_ops.cast(self.gain, dtype=dtype)
      if len(shape) == 3:
          weight = array_ops.scatter_nd([[(shape[0] - 1) // 2]],
                                          array_ops.expand_dims(q, 0), shape)
      elif len(shape) == 4:
          weight = array_ops.scatter_nd([[(shape[0] - 1) // 2,
                                          (shape[1] - 1) // 2]],
                                          array_ops.expand_dims(q, 0), shape)
      else:
          weight = array_ops.scatter_nd([[(shape[0] - 1) // 2, (shape[1] - 1) // 2,
                                          (shape[2] - 1) // 2]],
                                          array_ops.expand_dims(q, 0), shape)
      return weight

    def get_config(self):
        return {"gain": self.gain, "seed": self.seed, "dtype": self.dtype.name}



class ConvolutionOrthogonal(Initializer):
    """Initializer that generates orthogonal kernel for ConvNets.

    Base class used to construct 1D, 2D and 3D orthogonal kernels for convolution.

    Args:
        gain: multiplicative factor to apply to the orthogonal matrix. Default is 1.
            The 2-norm of an input is multiplied by a factor of `gain` after applying
            this convolution.
        seed: A Python integer. Used to create random seeds. See
            `tf.compat.v1.set_random_seed` for behavior.
             dtype: Default data type, used if no `dtype` argument is provided when
             calling the initializer. Only floating point types are supported.
    References:
        [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
        ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
    """

    def _assert_float_dtype(self, dtype):
          """Validate and return floating point type based on `dtype`.
        
          `dtype` must be a floating point type.
        
          Args:
            dtype: The data type to validate.
        
          Returns:
            Validated type.
        
          Raises:
            ValueError: if `dtype` is not a floating point type.
          """
          if not dtype.is_floating:
            raise ValueError("Expected floating point type, got %s." % dtype)
          return dtype

    def __init__(self, gain=1.0, seed=None, dtype=dtypes.float32):
        self.gain = gain
        self.dtype = self._assert_float_dtype(dtypes.as_dtype(dtype))
        self.seed = seed

    def __call__(self, shape, dtype=None, partition_info=None):
        raise NotImplementedError

    def get_config(self):
        return {"gain": self.gain, "seed": self.seed, "dtype": self.dtype.name}

    # Helper functions.
    def _orthogonal_matrix(self, n):
        """Construct an n x n orthogonal matrix.
    
        Args:
          n: Dimension.
    
        Returns:
          A n x n orthogonal matrix.
        """
        a = random_ops.random_normal([n, n], dtype=self.dtype, seed=self.seed)
        if self.seed:
          self.seed += 1
        q, r = gen_linalg_ops.qr(a)
        d = array_ops.diag_part(r)
        # make q uniform
        q *= math_ops.sign(d)
        return q

    def _symmetric_projection(self, n):
        """Compute a n x n symmetric projection matrix.
    
        Args:
          n: Dimension.
    
        Returns:
          A n x n symmetric projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
        """
        q = self._orthogonal_matrix(n)
        # randomly zeroing out some columns
        mask = math_ops.cast(
            random_ops.random_normal([n], seed=self.seed) > 0, self.dtype)
        if self.seed:
            self.seed += 1
        c = math_ops.multiply(q, mask) # why mask??
        
        return math_ops.matmul(c, array_ops.matrix_transpose(c))



class ConvolutionOrthogonal2D(ConvolutionOrthogonal):
    """Initializer that generates a 2D orthogonal kernel for ConvNets.

    The shape of the tensor must have length 4. The number of input
    filters must not exceed the number of output filters.
    The orthogonality(==isometry) is exact when the inputs are circular padded.
    There are finite-width effects with non-circular padding (e.g. zero padding).
    See algorithm 1 in (Xiao et al., 2018).

    Args:
        gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
          This has the effect of scaling the output 2-norm by a factor of `gain`.
        seed: A Python integer. Used to create random seeds. See
          `tf.compat.v1.set_random_seed` for behavior.
        dtype: Default data type, used if no `dtype` argument is provided when
          calling the initializer. Only floating point types are supported.
    References:
          [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
          ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
    """

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        if len(shape) != 4:
            raise ValueError("The tensor to initialize must be four-dimensional")
    
        if shape[-2] > shape[-1]:
            raise ValueError("In_filters cannot be greater than out_filters.")
    
        if shape[0] != shape[1]:
            raise ValueError("Kernel sizes must be equal.")
    
        kernel = self._orthogonal_kernel(shape[0], shape[2], shape[3])
        kernel *= math_ops.cast(self.gain, dtype=dtype)
        return kernel

    def _dict_to_tensor(self, x, k1, k2):
        """Convert a dictionary to a tensor.
    
        Args:
            x: A k1 * k2 dictionary.
            k1: First dimension of x.
            k2: Second dimension of x.
    
        Returns:
            A k1 * k2 tensor.
        """
        return array_ops.stack([array_ops.stack([x[i, j] for j in range(k2)])
                                for i in range(k1)])

    def _block_orth(self, p1, p2):
        """Construct a 2 x 2 kernel.
    
        Used to construct orthgonal kernel.
    
        Args:
            p1: A symmetric projection matrix.
            p2: A symmetric projection matrix.
    
        Returns:
            A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                          [(1-p1)p2, (1-p1)(1-p2)]].
        Raises:
            ValueError: If the dimensions of p1 and p2 are different.
        """
        if p1.shape.as_list() != p2.shape.as_list():
            raise ValueError("The dimension of the matrices must be the same.")
        n = p1.shape.as_list()[0]
        kernel2x2 = {}
        eye = linalg_ops_impl.eye(n, dtype=self.dtype)
        kernel2x2[0, 0] = math_ops.matmul(p1, p2)
        kernel2x2[0, 1] = math_ops.matmul(p1, (eye - p2))
        kernel2x2[1, 0] = math_ops.matmul((eye - p1), p2)
        kernel2x2[1, 1] = math_ops.matmul((eye - p1), (eye - p2))
    
        return kernel2x2

    def _matrix_conv(self, m1, m2):
        """Matrix convolution.
    
        Args:
            m1: A k x k dictionary, each element is a n x n matrix.
            m2: A l x l dictionary, each element is a n x n matrix.
    
        Returns:
            (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
        Raises:
            ValueError: if the entries of m1 and m2 are of different dimensions.
        """

        n = (m1[0, 0]).shape.as_list()[0]
        if n != (m2[0, 0]).shape.as_list()[0]: #cout
            raise ValueError("The entries in matrices m1 and m2 "
                             "must have the same dimensions!")
        k = int(np.sqrt(len(m1))) #nxn->n (n or m must be 2)
        l = int(np.sqrt(len(m2))) #mxm->m
        result = {}
        size = k + l - 1 #n+2+1, i.e., 3x3, 4x4, 5x5 for each iteration
        # Compute matrix convolution between m1 and m2.
        for i in range(size):
            for j in range(size):
                result[i, j] = array_ops.zeros([n, n], self.dtype)
                for index1 in range(min(k, i + 1)):
                    for index2 in range(min(k, j + 1)):
                        if (i - index1) < l and (j - index2) < l:
                            result[i, j] += math_ops.matmul(m1[index1, index2],
                                  m2[i - index1, j - index2])
        return result

    def _orthogonal_kernel(self, ksize, cin, cout):
        """Construct orthogonal kernel for convolution.
    
        Args:
            ksize: Kernel size.
            cin: Number of input channels.
            cout: Number of output channels.
    
        Returns:
            An [ksize, ksize, cin, cout] orthogonal kernel.
        Raises:
            ValueError: If cin > cout.
        """
        if cin > cout:
            raise ValueError("The number of input channels cannot exceed "
                             "the number of output channels.")
        orth = self._orthogonal_matrix(cout)[0:cin, :]
        if ksize == 1:
            return array_ops.expand_dims(array_ops.expand_dims(orth, 0), 0)
    
        p = self._block_orth(
                self._symmetric_projection(cout), self._symmetric_projection(cout))
        for _ in range(ksize - 2):
            temp = self._block_orth(
                    self._symmetric_projection(cout), self._symmetric_projection(cout))
            p = self._matrix_conv(p, temp)
        for i in range(ksize):
            for j in range(ksize):
                p[i, j] = math_ops.matmul(orth, p[i, j])
    
        return self._dict_to_tensor(p, ksize, ksize)


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
        elif sinitializer == 'orthogonal':
            init_fn = ConvolutionOrthogonal2D(gain=1.0)
        elif sinitializer == 'delta_orthogonal':
            init_fn = ConvolutionDeltaOrthogonal(gain=1.0)
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
            
            count = tf.get_variable(name='count', trainable=False, initializer=tf.constant(0.0))
            sh_x = tf.get_variable(name='sh_x', shape=(iCin), trainable=False, initializer=tf.constant_initializer(value=0.0))
            sh_x2 = tf.get_variable(name='sh_x2', shape=(iCin), trainable=False, initializer=tf.constant_initializer(value=0.0))
            
            #beta = tf.get_variable(name='beta', shape=(1), initializer=tf.constant_initializer(value=0.0))
            #gamma = tf.get_variable(name='gamma', shape=(1), initializer=tf.constant_initializer(value=1.0))
            
            if smode == 'CONV':
                               
                batch_mean, batch_var = tf.nn.moments(tinputs, [0,1,2], name='moments')
            
            elif smode == 'DENSE':
            
                batch_mean, batch_var = tf.nn.moments(tinputs, [0], name='moments')
            
            else:
                raise NotImplementedError('%s - mode of BN is not defined in %s'%(smode, sname))
            
            count, sh_x, sh_x2 = tf.cond(brst, lambda:(count.assign(1.0), sh_x.assign(batch_mean), sh_x2.assign(batch_var+batch_mean**2)), 
                                         lambda:(tf.cond(btrain, lambda:(count.assign_add(1.0), sh_x.assign_add(batch_mean), sh_x2.assign_add(batch_var+batch_mean**2)), 
                                                         lambda:(tf.identity(count), tf.identity(sh_x), tf.identity(sh_x2)))))
            #count, sh_x, sh_x2 = tf.cond(btrain, lambda:(count.assign_add(1.0), sh_x.assign_add(batch_mean), sh_x2.assign_add(batch_var + batch_mean**2)), lambda: (count, sh_x, sh_x2))
                        
            def mean_var_with_update():
                
                with tf.control_dependencies([count, sh_x, sh_x2]):
    
                    return tf.identity(batch_mean), tf.identity(batch_var)
                    #return ema.average(batch_mean)/denom, ema.average(batch_var)/denom
            
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
        #print(lfilter_shape)
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            # INITIALIZERS
            init_fn = self.initializer(sinitializer)
            kernels = []
            if sinitializer == "orthogonal":
                if lfilter_shape[-2] > lfilter_shape[-1]:
                    init_fn = self.initializer("he_normal")
                    kernels = tf.get_variable(name='kernel', shape=lfilter_shape, initializer=init_fn)
                else:
                    kernels = tf.get_variable(name='kernel', initializer=init_fn(lfilter_shape))
            elif sinitializer == 'delta_orthogonal':
                if lfilter_shape[-2] > lfilter_shape[-1]:
                    init_fn = self.initializer("he_normal")
                    kernels = tf.get_variable(name='kernel', shape=lfilter_shape, initializer=init_fn)
                else:
                    kernels = tf.get_variable(name='kernel', initializer=init_fn(lfilter_shape))
            else:
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
    def deconv(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sname=None):
        #print(lfilter_shape)
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            # INITIALIZERS
            init_fn = self.initializer(sinitializer)
            kernels = []
            if sinitializer == "orthogonal":
                if lfilter_shape[-1] > lfilter_shape[-2]:
                    init_fn = self.initializer("he_normal")
                    kernels = tf.get_variable(name='kernel', shape=lfilter_shape, initializer=init_fn)
                else:
                    shape = lfilter_shape[0:2] + (lfilter_shape[3], lfilter_shape[2])
                    tensor_k = tf.get_variable(name='kernel', initializer=init_fn(shape))
                    kernels = tf.transpose(tensor_k, perm=(0, 1, 3, 2))
            elif sinitializer == 'delta_orthogonal':
                if lfilter_shape[-1] > lfilter_shape[-2]:
                    init_fn = self.initializer("he_normal")
                    kernels = tf.get_variable(name='kernel', shape=lfilter_shape, initializer=init_fn)
                else:
                    shape = lfilter_shape[0:2] + (lfilter_shape[3], lfilter_shape[2])
                    tensor_k = tf.get_variable(name='kernel', initializer=init_fn(shape))
                    kernels = tf.transpose(tensor_k, perm=(0, 1, 3, 2))
            else:
                kernels = tf.get_variable(name='kernel', shape=lfilter_shape, initializer=init_fn)
            
            # ACCORDING TO THE GIVEN PADDING PROCESS
            if spadding == 'SAME':
                out_shape = tf.cast(tf.cast(tf.shape(tinputs), tf.float32) * (1., float(istride), float(istride), lfilter_shape[2]/lfilter_shape[3]) + 1e-3, dtype=tf.int32)
                x = tf.nn.conv2d_transpose(tinputs, kernels, strides=(1, istride, istride, 1), output_shape=out_shape, padding='SAME')
            elif spadding == 'REFLECT':
                
                pad1 = tf.cast(tf.subtract(lfilter_shape[0:2], 1)/2, dtype=tf.int32)
                pad2 = tf.cast(tf.cast(lfilter_shape[0:2], dtype=tf.float32)/2., dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
                tinputs = tf.pad(tinputs, pad_size, 'REFLECT')
                
                out_shape = tf.cast(tf.cast(tf.shape(tinputs), tf.float32) * (1., float(istride), float(istride), lfilter_shape[2]/lfilter_shape[3]) + 1e-3, dtype=tf.int32)
                t = tf.nn.conv2d_transpose(tinputs, kernels, strides=(1, istride, istride, 1), output_shape=out_shape, padding='SAME')
                
                h_st = pad1[0] * istride
                if pad2[0] == 0:
                    h_ed = None
                else:
                    h_ed = -pad2[0] * istride
                
                w_st = pad1[1] * istride
                if pad2[1] == 0:
                    w_ed = None
                else:
                    w_ed = -pad2[1] * istride
                                
                x = t[:,h_st:h_ed, w_st:w_ed, :]
                
            elif spadding == 'VALID':
                
                pad1 = tf.cast(tf.subtract(lfilter_shape[0:2], 1)/2, dtype=tf.int32)
                pad2 = tf.cast(tf.cast(lfilter_shape[0:2], dtype=tf.float32)/2., dtype=tf.int32)
                pad_size = [[0, 0], [pad1[0],pad2[0]], [pad1[1], pad2[1]], [0,0]]
                                
                out_shape = tf.cast(tf.cast(tf.shape(tinputs), tf.float32) * 
                                    (1., float(istride), float(istride), lfilter_shape[2]/lfilter_shape[3]) + ((0, pad1[0]+pad2[0], pad1[1]+pad2[1], 0) + 1e-3), dtype=tf.int32)
                t = tf.nn.conv2d_transpose(tinputs, kernels, strides=(1, istride, istride, 1), output_shape=out_shape, padding='SAME')
                
                h_st = pad1[0]
                if pad2[0] == 0:
                    h_ed = None
                else:
                    h_ed = -pad2[0]
                
                w_st = pad1[1]
                if pad2[1] == 0:
                    w_ed = None
                else:
                    w_ed = -pad2[1]
                                
                x = t[:,h_st:h_ed, w_st:w_ed, :]
                        
            if buse_bias:
                bias = tf.get_variable(name='bias', shape=(lfilter_shape[2]), initializer=tf.constant_initializer(value=0))
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
                slope = tf.get_variable(name='wd_slope', initializer=tf.constant(0.2))
                x = tf.where(tf.less(tinputs, 0.), tinputs*slope, tinputs)
            elif sactivation == 'SWISH':
                x = tf.nn.swish(tinputs)
            elif sactivation == 'DuLin':
                pslope = tf.get_variable(name='wd_pslope', initializer=tf.constant(1.0))
                nslope = tf.get_variable(name='wd_nslope', initializer=tf.constant(1.0))
                x = tf.where(tf.less(tinputs, 0.), tinputs*nslope, tinputs*pslope)
            elif sactivation == 'Sigmoid':
                x = tf.sigmoid(tinputs)
            else:
                raise NotImplementedError('sactivation parameter ({:s}) is not defined'.format(sactivation))
                
            return x


    @layer
    def convact(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sactivation='ReLu', sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            (self.feed(tinputs)
                 .conv(lfilter_shape=lfilter_shape, istride=istride, spadding=spadding, buse_bias=buse_bias, sinitializer=sinitializer, sname='conv')
                 .activation(sactivation=sactivation, sname='act'))
                            
            return self.terminals[0]
    
    @layer
    def deconvact(self, tinputs, lfilter_shape=(3,3,1,1), istride=1, spadding='SAME', buse_bias=False, sinitializer='he_normal', sactivation='ReLu', sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            (self.feed(tinputs)
                 .deconv(lfilter_shape=lfilter_shape, istride=istride, spadding=spadding, buse_bias=buse_bias, sinitializer=sinitializer, sname='deconv')
                 .activation(sactivation=sactivation, sname=sname+'act'))
                            
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
        ksize = [1,ipool_size,ipool_size,1]
        stride = [1,istride,istride,1]
        x = tf.nn.max_pool(tinputs, ksize, stride, spadding)
        #x = tf.nn.max_pool2d(tinputs, ipool_size, istride, spadding)
                                    
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
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            x = tf.cond(buse_drop, lambda: tf.nn.dropout(tinputs, rate=frate), lambda: tinputs, name='drop_cond')
        
            return x
        
    
    @layer
    def flatten(self, tinputs, sname=None):
                                                            
            with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
                return tf.reshape(tinputs, (tf.shape(tinputs)[0], -1), name='flt_reshape')
        
        
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
        
    @layer
    def depth_to_space(self, tinputs, iblk_size=2, sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            x = tf.nn.depth_to_space(tinputs, iblk_size, name='dep2sp')
            
            return x
        
    
    @layer
    def space_to_depth(self, tinputs, iblk_size=2, sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            x = tf.nn.space_to_depth(tinputs, iblk_size, name='sp2dep')
            
            return x
    
    
    @layer
    def to_reservoir(self, tinputs, sname=None):
        
        self.reservoir[sname] = tinputs
        
        return tinputs
    
    
    @layer
    def concat(self, tinputs, ltensors, axis=3, sname=None):
        
        value = [tinputs, ] + ltensors
        
        return tf.concat(value, axis=axis, name=sname)
    
    @layer
    def sp_attn(self, tinputs, tensor, lfilter_shape=(3,3,1,1), istride=1, spadding='REFLECT', buse_bias=False, sinitializer='delta_orthogonal', sname=None):
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            
            (self.feed(tinputs)
                 .conv(lfilter_shape=lfilter_shape, istride=istride, spadding=spadding, buse_bias=buse_bias, sinitializer=sinitializer, sname='conv')
                 .activation(sactivation='Sigmoid', sname='act'))
            
        return tinputs * self.terminals[0] + tensor * (1. - self.terminals[0])
        
    @layer
    def pypool(self, tinputs, iCin, smethod='add', sname=None):
        
        py_stages = [self.py_stage(tinputs, split_num=sn) for sn in [1, 2, 3, 6]]
                
        # 1x1 convolution for channel conversion 4*Cin to Cin
        features = tf.concat(py_stages, axis=3)
        
        with tf.variable_scope(sname, reuse=tf.AUTO_REUSE):
            (self.feed(features)
                 .convact(lfilter_shape=(1,1,4*iCin,iCin), istride=1, spadding='VALID', buse_bias=True, sinitializer='delta_orthogonal', sactivation='DuLin', sname='py_conv'))
            
        if smethod == 'add':
            out = tinputs + self.terminals[0]
        elif smethod == 'concat': 
            out = tf.concat((self.terminals[0], tinputs), axis=3)
        else:
            raise NotImplementedError('pypool parameter ({:s}) is not defined'.format(smethod))
    
        return out

    def py_stage(self, tinputs, split_num=1):
        
        h_size = tf.shape(tinputs)[1]/split_num # horizontal unit size
        w_size = tf.shape(tinputs)[2]/split_num # vertical unit size
        
        h=[tf.cast(h_size, tf.int32)] # first element
        w=[tf.cast(w_size, tf.int32)] # first element
        
        acc_h = h[0]
        acc_w = w[0]
        
        for idx in range(2,split_num+1):
            h.append(tf.cast(h_size*idx, tf.int32)-acc_h) # i.e., when split_num is 6, this gives 3, 4, 4, 3, 4, 4 splits
            acc_h += h[-1]
            
            w.append(tf.cast(w_size*idx, tf.int32)-acc_w)
            acc_w += w[-1]
            
        h[-1] += (tf.shape(tinputs)[1] - tf.reduce_sum(h))
        w[-1] += (tf.shape(tinputs)[2] - tf.reduce_sum(w))
                
        h_sp = tf.split(tinputs, h, axis=1) # horizontal split first
        
        h_merge = []
        for idx in range(len(h_sp)):
            w_sp = tf.split(h_sp[idx], w, axis=2) # vertical split
            # reduce mean for each split and concatenate them to width dim. to be an element of a list of N x 1 x split_dim x C
            h_merge.append(tf.concat([tf.reduce_mean(part, axis=(1,2), keepdims=True) for part in w_sp], axis=2)) 
            # [tf.reduce_mean(part, axis=(1,2), keepdims=True) for part in w_sp] => a list of reduce_mean values
            # to be concatenated by tf.concat("above list", axis=2)
            # placed into a list h_merge = [concatenate_for_h_sp[0], concatenate_for_sp[1], .... , concatenate_for_sp[split_num-1]]
                
        
        h_merge = tf.concat(h_merge, axis=1) # concatenate the h_merge to be N x split_dim x split_dim x C
        
        return tf.image.resize_bilinear(h_merge, tf.shape(tinputs)[1:3]) # 1x1 convolution before return this resized tensor

    
    
            
    