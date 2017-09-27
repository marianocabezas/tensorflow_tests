from operator import mul
import sys
import itertools
import tensorflow as tf
import numpy as np


def layer_from_dicts(layer_dict):
    class_name = layer_dict.pop('class')
    layer = getattr(sys.modules[__name__], class_name)(**layer_dict)

    return layer


class Tensor(object):

    counter = itertools.count()

    def __init__(self, node, input, output):
        self.input = input
        self.output = output
        self.node = node
        self.name = 'Input%d.T' % Tensor.counter.next() if self.node is None else node.name + '.T'


class Layer(object):
    activations = {
        'relu': tf.nn.relu
    }

    @staticmethod
    def _weight_variable(shape, scope='global'):
        try:
            with tf.variable_scope(scope):
                var = tf.get_variable('Weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        except ValueError:
            with tf.variable_scope(scope, reuse=True):
                var = tf.get_variable('Weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return var

    @staticmethod
    def _bias_variable(shape, scope='global'):
        try:
            with tf.variable_scope(scope):
                var = tf.get_variable('Bias', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        except ValueError:
            with tf.variable_scope(scope, reuse=True):
                var = tf.get_variable('Bias', shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        return var

    def __init__(self, name='layer'):
        self.name = name
        self.trainable = True
        self.W = None
        self.b = None

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

    def set_trainable(self, trainable):
        self.trainable = trainable

    def get_config(self):
        return {'layer': None, 'name': self.name}

    def print_info(self):
        weights_s = 'weights %r, ' % repr(self.W) if self.W is not None else 'no weights, '
        bias_s = 'bias %r' % repr(self.b) if self.b is not None else 'no bias'
        print('%s: %s%s' % (self.name, weights_s, bias_s))


def Input(shape):
    return Tensor(None, None, tf.placeholder(tf.float32, [None] + shape))


class Reshape(Layer):

    counter = itertools.count()

    def __init__(self, shape, name=None):
        if name is None:
            name = 'reshape%02d' % Reshape.counter.next()
        super(Reshape, self).__init__(name=name)
        self.shape = shape

    def __call__(self, x):
        return Tensor(self, x, tf.reshape(x.output, self.shape))

    def get_config(self):
        return {
            'class': self.__class__.__name__,
            'shape': self.shape,
            'name': self.name
        }


class Concatenate(Layer):

    counter = itertools.count()

    def __init__(self, axis=0, name=None):
        if name is None:
            name = 'concatenate%02d' % Reshape.counter.next()
        super(Concatenate, self).__init__(name=name)
        self.axis = axis

    def __call__(self, x):
        return Tensor(self, x, tf.concat(x, axis=self.axis))

    def get_config(self):
        return {
            'class': self.__class__.__name__,
            'axis': self.axis,
            'name': self.name
        }


class Add(Layer):

    counter = itertools.count()

    def __init__(self, axis=0, name=None):
        if name is None:
            name = 'concatenate%02d' % Reshape.counter.next()
        super(Add, self).__init__(name=name)
        self.axis = axis

    def __call__(self, x):
        return Tensor(self, x, tf.add_n(x))

    def get_config(self):
        return {
            'class': self.__class__.__name__,
            'axis': self.axis,
            'name': self.name
        }


class Conv(Layer):

    counter = itertools.count()

    def __init__(
            self,
            filters,
            kernel_size,
            strides=None,
            padding='valid',
            activation=None,
            name=None
    ):
        assert 0 < len(kernel_size) < 4, 'The kernel size is not supported: %r' % repr(kernel_size)
        if name is None:
            name = 'conv%02d' % Conv.counter.next()
        super(Conv, self).__init__(name=name)
        self.kernel_size = kernel_size
        self.rank = len(self.kernel_size)
        self.filters = filters
        if strides is None:
            strides = [1] * self.rank
        self.strides = [1] + strides + [1]
        self.padding = padding
        self.activation = activation

    def __call__(self, x):
        input_shape = x.output.get_shape().as_list()
        weights_shape = list(self.kernel_size) + input_shape[-1:] + [self.filters]
        self.W = Layer._weight_variable(shape=weights_shape, scope=self.name)
        self.b = Layer._bias_variable([self.filters], scope=self.name)
        conv_f = [tf.nn.conv1d, tf.nn.conv2d, tf.nn.conv3d]
        conv = conv_f[self.rank-1](x.output, self.W, strides=self.strides, padding=self.padding) + self.b
        if self.activation is not None:
            conv = Layer.activations[self.activation](conv)
        return Tensor(self, x, conv)

    def get_config(self):
        config = {
            'class': self.__class__.__name__,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides[1:-1],
            'padding': self.padding,
            'activation': self.activation,
            'name': self.name
        }
        return config


class MaxPool(Layer):

    counter = itertools.count()

    def __init__(self, pool_size, padding='valid', name=None):
        if name is None:
            name = 'max_pool%02d' % MaxPool.counter.next()
        super(MaxPool, self).__init__(name=name)
        self.pool_size = pool_size
        self.kernel_size = [1] + list(pool_size) + [1]
        self.strides = [1] + list(pool_size) + [1]
        self.padding = padding

    def __call__(self, x):
        return Tensor(self, x, tf.nn.max_pool(
            x.output,
            ksize=self.kernel_size,
            strides=self.strides,
            padding=self.padding
        ))

    def get_config(self):
        config = {
            'class': self.__class__.__name__,
            'pool_size': self.pool_size,
            'padding': self.padding,
            'name': self.name
        }
        return config


class Dense(Layer):

    counter = itertools.count()

    def __init__(self, units, activation=None, name=None):
        if name is None:
            name = 'dense%02d' % Dense.counter.next()
        super(Dense, self).__init__(name=name)
        self.units = units
        self.activation = activation

    def __call__(self, x):
        input_shape = x.output.get_shape().as_list()
        self.W = Layer._weight_variable([reduce(mul, input_shape[1:]), self.units], scope=self.name)
        self.b = Layer._bias_variable([self.units], scope=self.name)
        x_flat = tf.reshape(x.output, [-1, reduce(mul, input_shape[1:])])
        fc = tf.nn.relu(tf.matmul(x_flat, self.W) + self.b)
        if self.activation is not None:
            fc = Layer.activations[self.activation](fc)
        return Tensor(self, x, fc)

    def get_config(self):
        config = {
            'class': self.__class__.__name__,
            'units': self.units,
            'activation': self.activation,
            'name': self.name
        }
        return config


class Dropout(Layer):

    counter = itertools.count()

    def __init__(self, rate, seed=None, name=None):
        if name is None:
            name = 'drop%02d' % Dropout.counter.next()
        super(Dropout, self).__init__(name=name)
        self.rate = rate
        if seed is None:
            seed = np.random.randint(10000)
        self.seed = seed

    def __call__(self, x):
        self.input = x
        retain_prob = 1. - self.rate
        return Tensor(self, x, tf.nn.dropout(self.input.output * 1., retain_prob, seed=self.seed))

    def get_config(self):
        config = {
            'class': self.__class__.__name__,
            'rate': self.rate,
            'seed': self.seed,
            'name': self.name
        }
        return config
