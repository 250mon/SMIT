import tensorflow.compat.v1 as tf
import tensorflow.bitwise as tw
tf.disable_eager_execution()

def my_sparse_to_dense(arr, output_shape):
    # 1 dim
    hist = tf.bincount(arr)
    # 2 dim (tf.where() returns a Tensor of shape (num_true, rank(condition)), so (2, 1) where '1' holds the index)
    nonzero_indices = tf.where(tf.not_equal(hist, 0))
    # 2 dim
    nonzero_values_2d = tf.gather(hist, nonzero_indices)
    # 1 dim
    nonzero_values_1d = tf.squeeze(nonzero_values_2d)
    # sparse_indices: 0-D, 1-D, or 2-D Tensor of type int32 or int64.
    #      sparse_indices[i] contains the complete index where sparse_values[i] will be placed.
    # output_shape: A 1-D Tensor of the same type as sparse_indices.
    # sparse_values: A 0-D or 1-D Tensor.
    #      Values corresponding to each row of sparse_indices, or a scalar value to be used for all sparse indices.
    # 1 dim
    conf_matrix = tf.sparse_to_dense(nonzero_indices,
                                     output_shape,
                                     nonzero_values_1d,
                                     0)

    sess = tf.Session()
    result = sess.run(conf_matrix)
    return result


if __name__ == '__main__':
    two_d_arr = tf.constant([[0x0000, 0x0003], [0x0002, 0x0002]])
    print(my_sparse_to_dense(two_d_arr, (4,)))
