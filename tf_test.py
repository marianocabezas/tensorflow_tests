from __future__ import print_function
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from utils import color_codes
from operator import mul


activations = {
    'relu': tf.nn.relu
}


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x,  filters, kernel_size=(3, 3), strides=list([1, 1]), padding='SAME', activation=None):
    W_conv = weight_variable(list(kernel_size) + x.get_shape().as_list()[-1:] + [filters])
    b_conv = bias_variable([filters])
    strides = [1] + strides + [1]
    conv = tf.nn.conv2d(x, W_conv, strides=strides, padding=padding) + b_conv
    return conv


def max_pool_2d(x, pool_size=(2, 2), padding='SAME'):
    kernel_size = [1] + list(pool_size) + [1]
    strides = [1] + list(pool_size) + [1]
    return tf.nn.max_pool(x, ksize=kernel_size, strides=strides, padding=padding)


def dense(x, units, activation=None):
    W_fc = weight_variable([reduce(mul, x.get_shape().as_list()[1:]), units])
    b_fc = bias_variable([units])
    x_flat = tf.reshape(x, [-1, reduce(mul, x.get_shape().as_list()[1:])])
    fc = tf.nn.relu(tf.matmul(x_flat, W_fc) + b_fc)
    if activation is not None:
        fc = activations[activation](fc)
    return fc


def dropout(x, rate):
    keep_prob = tf.Variable(tf.constant(rate))
    # keep_prob = tf.placeholder(tf.float32)
    return tf.nn.dropout(x, keep_prob)


def print_metrics(i, train_losses, train_accs, val_losses, val_accs):
    def print_metric(i, metric, color, min):
        assert isinstance(metric, list), '%r is not a tuple (optimum, current, best_i)' % metric
        assert isinstance(color, basestring), '%r is not a formatting string' % color
        is_better = metric[1] < metric[0] if min else metric[1] > metric[0]
        metric_s = '%f' % metric[1]
        if is_better:
            metric[0] = metric[1]
            metric[2] = i
            metric_s = color + metric_s + color_codes()['nc']
        print(metric_s, end='\t')

    c = color_codes()

    # Structure of losses/acc:
    # 'name': (min/max, current, best_i)
    metrics = zip(
        sorted(train_losses.iteritems()),
        sorted(train_accs.iteritems()),
        sorted(val_losses.iteritems()),
        sorted(val_accs.iteritems())
    )
    print(''.join([' ']*130), end='\r')
    sys.stdout.flush()
    print('Epoch %03d' % i, end='\t')
    for ((k_tl, v_tl), (k_ta, v_ta), (k_vl, v_vl), (k_va, v_va)) in metrics:
        print_metric(i, v_tl, c['c'], True)
        print_metric(i, v_ta, c['c'], False)
        print_metric(i, v_vl, c['g'], True)
        print_metric(i, v_va, c['g'], False)
        print()


def print_current(epoch, step, n_batches, curr_values):
    percent = 20 * step / n_batches
    bar = '[' + ''.join([' '] * percent) + '>' + ''.join(['-'] * (20 - percent)) + ']'
    curr_values_s = ' train_loss %f (%f) train_acc %f (%f)' % curr_values
    print('Epoch %03d\t(%d/%d) ' % (epoch, step, n_batches) + bar + curr_values_s, end='\r')
    sys.stdout.flush()


def train_loop(
        input,
        output,
        tr_data,
        tr_labels,
        val_data,
        val_labels,
        session,
        train_step,
        loss_f,
        acc_f,
        n_iters,
        batch_size
):
    n_batches = -(-len(tr_data) / batch_size)
    train_loss = {'train_loss': [np.inf, np.inf, 0]}
    train_acc = {'train_acc': [-np.inf, -np.inf, 0]}
    val_loss = {'val_loss': [np.inf, np.inf, 0]}
    val_acc = {'val_acc': [-np.inf, -np.inf, 0]}

    print(''.join([' ']*9) + '\ttrain_loss\ttrain_acc\tval_loss\tval_acc')
    for i in range(n_iters):
        idx = np.random.permutation(len(tr_data))
        x = tr_data[idx, :]
        y = tr_labels[idx, :]
        acc_sum = 0
        loss_sum = 0
        for step in range(n_batches):
            step_init = step*batch_size
            step_end = step*batch_size+batch_size
            batch_xs, batch_ys = x[step_init:step_end, :], y[step_init:step_end, :]
            session.run(train_step, feed_dict={input: batch_xs, output: batch_ys})
            curr_acc = session.run(acc_f, feed_dict={input: batch_xs, output: batch_ys})
            curr_loss = session.run(loss_f, feed_dict={input: batch_xs, output: batch_ys})
            acc_sum += curr_acc
            loss_sum += curr_loss
            curr_values = (curr_loss, loss_sum/(step+1), curr_acc,  acc_sum/(step+1))
            print_current(i, step, n_batches, curr_values)

        train_loss['train_loss'][1] = loss_sum / n_batches
        train_acc['train_acc'][1] = acc_sum / n_batches
        val_loss['val_loss'][1] = session.run(loss_f, feed_dict={input: val_data, output: val_labels})
        val_acc['val_acc'][1] = session.run(acc_f, feed_dict={input: val_data, output: val_labels})
        print_metrics(i, train_loss, train_acc, val_loss, val_acc)


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = conv2d(x_image, filters=32, kernel_size=(5, 5), activation='relu')
    h_pool1 = max_pool_2d(h_conv1)
    h_conv2 = conv2d(h_pool1, filters=64, kernel_size=(5, 5), activation='relu')
    h_pool2 = max_pool_2d(h_conv2)
    h_fc1 = dense(h_pool2, 1024, activation='relu')
    h_drop = dropout(h_fc1, 0.5)
    y_conv = dense(h_drop, 10)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    train_loop(
        x,
        y_,
        mnist.train.images,
        mnist.train.labels,
        mnist.test.images,
        mnist.test.labels,
        sess,
        train_step,
        cross_entropy,
        accuracy,
        100,
        1024
    )


if __name__ == '__main__':
    # Tensorflow config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    tf.logging.set_verbosity(tf.logging.INFO)

    main()

