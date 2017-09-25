from operator import mul
import itertools
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Layer(object):
    activations = {
        'relu': tf.nn.relu
    }

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __init__(self, layer=None, name='layer'):
        self.name = name
        self.output = None
        self.trainable = True
        if layer is None:
            self.W = None
            self.b = None
        else:
            self.W = layer.W
            self.b = layer.b

    def get_trainable_parameters(self):
        parameters = list()
        if self.trainable:
            if self.W is not None:
                parameters.append(self.W)
            if self.b is not None:
                parameters.append(self.b)
        return parameters

    def count_trainable_parameters(self):
        parameters = 0
        if self.trainable:
            if self.W is not None:
                parameters += reduce(mul, self.W.get_shape().as_list())
            if self.b is not None:
                parameters += reduce(mul, self.b.get_shape().as_list())
        return parameters

    def count_nontrainable_parameters(self):
        return 0

    def get_name(self):
        return self.name

    def set_trainable(self, trainable):
        self.trainable = trainable

    def get_output(self):
        return self.output

    def print_info(self):
        weights_s = 'weights %r, ' % repr(self.W) if self.W is not None else 'no weights, '
        bias_s = 'bias %r' % repr(self.b) if self.b is not None else 'no bias'
        print('%s: %s%s' % (self.name, weights_s, bias_s))


class Input(Layer):

    counter = itertools.count()

    def __init__(self, shape, name=None):
        if name is None:
            name = 'input%02d' % Input.counter.next()
        super(Input, self).__init__(name=name)
        self.output = tf.placeholder(tf.float32, [None] + shape)


class Reshape(Layer):

    counter = itertools.count()

    def __init__(self, x, shape, name=None):
        if name is None:
            name = 'reshape%02d' % Reshape.counter.next()
        super(Reshape, self).__init__(name=name)
        self.input = x
        self.output = tf.reshape(x.output, shape)


class Conv(Layer):

    counter = itertools.count()

    def __init__(
            self,
            x,
            filters,
            kernel_size,
            strides=None,
            layer=None,
            padding='SAME',
            activation=None,
            name=None
    ):
        assert 0 < len(kernel_size) < 4, 'The kernel size is not supported: %r' % repr(kernel_size)
        if name is None:
            name = 'conv%02d' % Conv.counter.next()
        self.input = x
        input_shape = self.input.output.get_shape().as_list()
        rank = len(kernel_size)
        if layer is None:
            super(Conv, self).__init__(name=name)
            self.W = Layer._weight_variable(list(kernel_size) + input_shape[-1:] + [filters])
            self.b = Layer._bias_variable([filters])
            self.kernel_size = kernel_size
            if strides is None:
                strides = [1] * rank
            self.strides = [1] + strides + [1]
            self.padding = padding
        else:
            super(Conv, self).__init__(layer=layer, name=name)
            self.kernel_size = layer.kernel_size
            self.strides = layer.strides
            self.padding = layer.padding
        conv_f = [tf.nn.conv1d, tf.nn.conv2d, tf.nn.conv3d]
        conv = conv_f[rank-1](self.input.output, self.W, strides=self.strides, padding=self.padding) + self.b
        if activation is not None:
            conv = Layer.activations[activation](conv)
        self.output = conv


class MaxPool(Layer):

    counter = itertools.count()

    def __init__(self, x, pool_size=None, padding='SAME', name=None):
        if name is None:
            name = 'max_pool%02d' % MaxPool.counter.next()
        super(MaxPool, self).__init__(name=name)
        self.input = x
        rank = len(self.input.output.get_shape().as_list()) - 2
        if pool_size is None:
            pool_size = [2] * rank
        self.kernel_size = [1] + list(pool_size) + [1]
        self.strides = [1] + list(pool_size) + [1]
        self.padding = padding
        self.output = tf.nn.max_pool(
            self.input.output,
            ksize=self.kernel_size,
            strides=self.strides,
            padding=self.padding
        )


class Dense(Layer):

    counter = itertools.count()

    def __init__(self, x, units, activation=None, name=None):
        if name is None:
            name = 'dense%02d' % Dense.counter.next()
        super(Dense, self).__init__(name=name)
        self.input = x
        input_shape = self.input.output.get_shape().as_list()
        self.W = Layer._weight_variable([reduce(mul, input_shape[1:]), units])
        self.b = Layer._bias_variable([units])
        x_flat = tf.reshape(self.input.output, [-1, reduce(mul, input_shape[1:])])
        fc = tf.nn.relu(tf.matmul(x_flat, self.W) + self.b)
        if activation is not None:
            fc = Layer.activations[activation](fc)
        self.output = fc


class Dropout(Layer):

    counter = itertools.count()

    def __init__(self, x, rate, seed=None, name=None):
        if name is None:
            name = 'drop%02d' % Dropout.counter.next()
        super(Dropout, self).__init__(name=name)
        self.input = x
        retain_prob = 1. - rate
        if seed is None:
            seed = np.random.randint(10000)
        self.output = tf.nn.dropout(self.input.output * 1., retain_prob, seed=seed)


def input_layer(shape):
    input_l = tf.placeholder(tf.float32, [None] + shape)
    return input_l


def conv(x, filters, kernel_size, strides, padding='SAME', activation=None):
    assert 0 < len(kernel_size) < 4, 'The kernel size is not supported: %r' % repr(kernel_size)
    conv_f = [conv1d, conv2d, conv3d]
    return conv_f[len(kernel_size)-1](x, filters, kernel_size, strides, padding, activation)


def conv1d(x,  filters, kernel_size=(3,), strides=list([1,]), padding='SAME', activation=None):
    W_conv = weight_variable(list(kernel_size) + x.get_shape().as_list()[-1:] + [filters])
    b_conv = bias_variable([filters])
    strides = [1] + strides + [1]
    conv = tf.nn.conv1d(x, W_conv, stride=strides, padding=padding) + b_conv
    if activation is not None:
        conv = Layer.activations[activation](conv)
    return conv


def conv2d(x, filters, kernel_size=(3, 3), strides=list([1, 1]), padding='SAME', activation=None):
    W_conv = weight_variable(list(kernel_size) + x.get_shape().as_list()[-1:] + [filters])
    b_conv = bias_variable([filters])
    strides = [1] + strides + [1]
    conv = tf.nn.conv2d(x, W_conv, strides=strides, padding=padding) + b_conv
    if activation is not None:
        conv = Layer.activations[activation](conv)
    return conv


def conv3d(x,  filters, kernel_size=(3, 3, 3), strides=list([1, 1, 1]), padding='SAME', activation=None):
    W_conv = weight_variable(list(kernel_size) + x.get_shape().as_list()[-1:] + [filters])
    b_conv = bias_variable([filters])
    strides = [1] + strides + [1]
    conv = tf.nn.conv3d(x, W_conv, strides=strides, padding=padding) + b_conv
    if activation is not None:
        conv = Layer.activations[activation](conv)
    return conv


def max_pool(x, pool_size, padding='SAME'):
    kernel_size = [1] + list(pool_size) + [1]
    strides = [1] + list(pool_size) + [1]
    return tf.nn.max_pool(x, ksize=kernel_size, strides=strides, padding=padding)


def max_pool_2d(x, pool_size=(2, 2), padding='SAME'):
    return max_pool(x, pool_size, padding)


def max_pool_3d(x, pool_size=(2, 2, 2), padding='SAME'):
    return max_pool(x, pool_size, padding)


def dense(x, units, activation=None):
    W_fc = weight_variable([reduce(mul, x.get_shape().as_list()[1:]), units])
    b_fc = bias_variable([units])
    x_flat = tf.reshape(x, [-1, reduce(mul, x.get_shape().as_list()[1:])])
    fc = tf.nn.relu(tf.matmul(x_flat, W_fc) + b_fc)
    if activation is not None:
        fc = Layer.activations[activation](fc)
    return fc


def dropout(x, rate, seed=None):
    retain_prob = 1. - rate
    if seed is None:
        seed = np.random.randint(10e6)
    return tf.nn.dropout(x * 1., retain_prob, seed=seed)