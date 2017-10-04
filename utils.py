from __future__ import print_function
import sys
from math import floor
import numpy as np
from scipy import ndimage as nd
import tensorflow as tf


def color_codes():
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
    }
    return codes


def print_headers(other_accs=list()):
    print(''.join([' '] * 3), end='\t')
    print('train_loss', end='\t')
    print('train_acc ', end='\t')
    print('val_loss  ', end='\t')
    print('val_acc   ', end='\t')
    for name in other_accs:
            print(name, end='\t')
    print()


def print_metrics(i, train_losses, train_accs, val_losses, val_accs, time):
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
    print(''.join([' ']*130), end='\r')
    sys.stdout.flush()
    print('%03d' % i, end='\t')
    print_metric(i, train_losses['train_loss'], c['c'], True)
    print_metric(i, train_accs['train_acc'], c['c'], False)
    print_metric(i, val_losses['val_loss'], c['g'], True)
    print_metric(i, val_accs['val_acc'], c['g'], False)
    other_accs = dict((k, v) for k, v in val_accs.items() if k is not 'val_acc')
    for v in other_accs.values():
        print_metric(i, v, c['g'], False)
    print('%fs' % time)


def print_current(epoch, step, n_batches, curr_values):
    percent = 20 * step / n_batches
    bar = '[' + ''.join([' '] * percent) + '>' + ''.join(['-'] * (20 - percent)) + ']'
    curr_values_s = ' train_loss %f (%f) train_acc %f (%f)' % curr_values
    print('%03d\t(%d/%d) ' % (epoch, step, n_batches) + bar + curr_values_s, end='\r')
    sys.stdout.flush()


def train_test_split(data, labels, test_size=0.1, random_state=42):
    # Init (Set the random seed and determine the number of cases for test)
    n_test = int(floor(data.shape[0]*test_size))

    # We create a random permutation of the data
    # First we permute the data indices, then we shuffle the data and labels
    np.random.seed(random_state)
    indices = np.random.permutation(range(0, data.shape[0]))
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    x_train = shuffled_data[:-n_test]
    x_test = shuffled_data[-n_test:]
    y_train = shuffled_labels[:-n_test]
    y_test = shuffled_data[-n_test:]

    return x_train, x_test, y_train, y_test


def val_split(data, val_rate=0.1, random_state=42):
    # Init (Set the random seed and determine the number of cases for test)
    n_test = int(floor(data.shape[0]*val_rate))

    # We create a random permutation of the data
    # First we permute the data indices, then we shuffle the data and labels
    np.random.seed(random_state)
    indices = np.random.permutation(range(0, data.shape[0]))
    shuffled_data = data[indices]

    train_data = shuffled_data[:-n_test]
    val_data = shuffled_data[-n_test:]

    return train_data, val_data


def leave_one_out(data_list, labels_list):
    for i in range(0, len(data_list)):
        yield data_list[:i] + data_list[i+1:], labels_list[:i] + labels_list[i+1:], i


def nfold_cross_validation(data_list, labels_list, n=5, random_state=42, val_data=None):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(xrange(len(data_list)))

    for i in xrange(n):
        indices = shuffled_indices[i::n]
        tst_data = data_list[indices]
        tst_labels = labels_list[indices]
        tr_labels = labels_list[[idx for idx in shuffled_indices if idx not in indices]]
        tr_data = data_list[[idx for idx in shuffled_indices if idx not in indices]]
        val_len = int(len(tr_data) * val_data) if val_data is not None else None
        if val_data is not None:
            yield tr_data[val_len:], tr_labels[val_len:], tr_data[:val_len], tr_labels[:val_len], tst_data, tst_labels
        else:
            yield tr_data, tr_labels, tst_data, tst_labels


def get_biggest_region(labels, opening=False):
    nu_labels = np.copy(labels)
    bin_mask = labels.astype(dtype=np.bool)
    if opening:
        strel = nd.morphology.iterate_structure(nd.morphology.generate_binary_structure(3, 3), 5)
        bin_op_mask = nd.morphology.binary_opening(bin_mask, strel)
        if np.count_nonzero(bin_op_mask) > 0:
            bin_mask = bin_op_mask
    if np.count_nonzero(bin_mask) > 0:
        blobs, _ = nd.measurements.label(bin_mask, nd.morphology.generate_binary_structure(3, 3))
        big_region = np.argmax(np.bincount(blobs.ravel())[1:])
        nu_labels[blobs != big_region + 1] = 0
    return nu_labels


def get_patient_info(p):
    p_name = '-'.join(p[0].rsplit('/')[-1].rsplit('.')[0].rsplit('-')[:-1])
    patient_path = '/'.join(p[0].rsplit('/')[:-1])

    return p_name, patient_path


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
