import tensorflow as tf


# TODO: Implement all the important measures (including Wasserstein)
def cross_entropy(x, y):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x))
    return cross_entropy


def accuracy(x, y):
    correct_prediction = tf.equal(tf.argmax(x, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy