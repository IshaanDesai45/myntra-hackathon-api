import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

def conv2d(input_, output, kernel, stride, relu, bn, name, stddev=0.01):
    with tf.compat.v1.variable_scope(name) as scope:
    # Convolution for a given input and kernel
        shape = [kernel, kernel, input_.get_shape()[-1], output]
        w = tf.compat.v1.get_variable('w', shape, initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input=input_, filters=w, strides=[1, stride, stride, 1], padding='SAME')
        # Add the biases
        b = tf.compat.v1.get_variable('b', [output], initializer=tf.compat.v1.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        if bn:
            conv = tf.compat.v1.layers.batch_normalization(conv)
        # ReLU non-linearity
        if relu:
            conv = tf.nn.relu(conv, name=scope.name)
        return conv

def max_pool(input_, kernel, stride, name):
    return tf.nn.max_pool2d(input=input_, ksize=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)

def linear(input_, output, name, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()
    with tf.compat.v1.variable_scope(name) as scope:
        matrix = tf.compat.v1.get_variable("Matrix", [shape[1], output], tf.float32,
                                 tf.compat.v1.random_normal_initializer(stddev=stddev))
        bias = tf.compat.v1.get_variable("bias", [output], initializer=tf.compat.v1.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias

def atrous_conv2d(input_, output, kernel, rate, relu, name, stddev=0.01):
    with tf.compat.v1.variable_scope(name) as scope:
    # Dilation convolution for a given input and kernel
        shape = [kernel, kernel, input_.get_shape()[-1], output]
        w = tf.compat.v1.get_variable('w', shape, initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.atrous_conv2d(input_, w, rate, padding='SAME')
        # Add the biases
        b = tf.compat.v1.get_variable('b', [output], initializer=tf.compat.v1.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        # ReLU non-linearity
        if relu:
            conv = tf.nn.relu(conv, name=scope.name)
        return conv
