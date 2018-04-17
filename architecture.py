import tensorflow as tf


def batch_norm(input, name="batch_norm"):
	with tf.variable_scope(name) as scope:
		input = tf.identity(input)
		channels = input.get_shape()[3]

		offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))

		mean, variance = tf.nn.moments(input, axes=[0,1,2], keep_dims=False)

		normalized_batch = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=1e-5)

		return normalized_batch 


def linear(input, output_size, name="linear"):
	shape = input.get_shape().as_list()

	with tf.variable_scope(name) as scope:
		matrix = tf.get_variable("W", [shape[1], output_size], tf.float32, tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))

		return tf.matmul(input, matrix) + bias, matrix


def conv2d(input, input_filters, output_filters, kernel, strides, padding = 'SAME', mode='CONSTANT', name='conv'):
    with tf.variable_scope(name) as scope:
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.get_variable("weights", shape, initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input, weight, strides=[1, strides, strides, 1], padding='SAME', name=name)

        return conv


def max_pool(input, kernel, stride, padding='SAME', name='pool'):
	return tf.nn.max_pool(input, 
						  ksize=[1,kernel,kernel,1],
						  strides=[1,stride,stride,1],
						  padding=padding,
						  name=name
						  )