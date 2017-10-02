from __future__ import print_function
import sys
import os
import argparse
from time import strftime
import numpy as np
from nibabel import load as load_nii
from tensorflow.examples.tutorials.mnist import input_data
from models import Model, save_model, load_model
from layers import Input, Reshape, Conv, MaxPool, Dense, Dropout, Concatenate, Permute, Activation, Flatten
from data_creation import get_cnn_centers, load_patches_train, get_mask_voxels, load_norm_list, get_patches_list
from data_manipulation.metrics import dsc_seg
from utils import color_codes, get_biggest_region


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--training-folder', dest='dir_train', default='/home/mariano/DATA/Brats17Test-Training/')
    parser.add_argument('-F', '--test-folder', dest='dir_test', default='/home/mariano/DATA/Brats17Test/')
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=17)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=4)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=512)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--down-factor', dest='dfactor', type=int, default=200)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=10)
    parser.add_argument('-E', '--epochs-repetition', action='store', dest='r_epochs', type=int, default=20)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=10)
    parser.add_argument('-v', '--validation-rate', action='store', dest='val_rate', type=float, default=0.25)
    parser.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
    parser.add_argument('-s', '--sequential', action='store_true', dest='sequential', default=False)
    parser.add_argument('-r', '--recurrent', action='store_true', dest='recurrent', default=False)
    parser.add_argument('-p', '--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('-P', '--patience', dest='patience', type=int, default=5)
    parser.add_argument('--flair', action='store', dest='flair', default='_flair.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='_t1.nii.gz')
    parser.add_argument('--t1ce', action='store', dest='t1ce', default='_t1ce.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='_t2.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='_seg.nii.gz')
    return vars(parser.parse_args())


def get_names_from_path(options, train=True):
    path = options['dir_train'] if train else options['dir_test']

    directories= filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])
    patients = sorted(directories)

    # Prepare the names
    flair_names = [os.path.join(path, p, p.split('/')[-1] + options['flair']) for p in patients]
    t2_names = [os.path.join(path, p, p.split('/')[-1] + options['t2']) for p in patients]
    t1_names = [os.path.join(path, p, p.split('/')[-1] + options['t1']) for p in patients]
    t1ce_names = [os.path.join(path, p, p.split('/')[-1] + options['t1ce']) for p in patients]

    label_names = np.array([os.path.join(path, p, p.split('/')[-1] + options['labels']) for p in patients])
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names


def get_brats_net(input_shape, filters_list, kernel_size_list, dense_size, nlabels):
    inputs = Input(shape=input_shape)
    conv = inputs
    for filters, kernel_size in zip(filters_list, kernel_size_list):
        conv = Conv(filters, kernel_size=(kernel_size,)*3, activation='relu', data_format='channels_first')(conv)

    full = Conv(dense_size, kernel_size=(1, 1, 1), data_format='channels_first', name='fc_dense', activation='relu')(conv)
    full = Conv(nlabels, kernel_size=(1, 1, 1), data_format='channels_first', name='fc')(full)

    rf = Concatenate(axis=1)([conv, full])

    rf_num = 1
    while np.product(rf.shape[2:]) > 1:
        rf = Conv(dense_size, kernel_size=(3, 3, 3), data_format='channels_first', name='rf_%d' % rf_num)(rf)
        rf_num += 1

    full = Reshape((nlabels, -1))(full)
    full = Permute((2, 1))(full)
    full_out = Activation('softmax', name='fc_out')(full)

    combo = Concatenate(axis=1)([Flatten()(conv), Flatten()(rf)])

    tumor = Dense(nlabels, activation='softmax', name='tumor')(combo)

    outputs = [tumor, full_out]
    # outputs = tumor

    net = Model(
        inputs=inputs,
        outputs=outputs,
        optimizer='adadelta',
        loss='categorical_cross_entropy',
        metrics='accuracy'
    )

    return net


def train_net(net, net_name, nlabels):
    options = parse_inputs()
    c = color_codes()
    # Data stuff
    train_data, train_labels = get_names_from_path(options)
    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    conv_blocks = options['conv_blocks']
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width] * conv_blocks
    balanced = options['balanced']
    val_rate = options['val_rate']
    preload = options['preload']
    fc_width = patch_width - sum(kernel_size_list) + conv_blocks
    fc_shape = (fc_width,) * 3

    try:
        net = load_model(net_name + '.mod')
    except IOError:
        for _ in range(options['r_epochs']):
            train_centers = get_cnn_centers(train_data[:, 0], train_labels, balanced=balanced)
            print(' '.join([''] * 16) + c['g'] + 'Loading data ' +
                  c['b'] + '(%d centers)' % (len(train_centers) / dfactor) + c['nc'])
            x, y = load_patches_train(
                image_names=train_data,
                label_names=train_labels,
                centers=train_centers,
                size=patch_size,
                fc_shape=fc_shape,
                nlabels=nlabels,
                dfactor=dfactor,
                preload=preload,
            )

            print(' '.join([''] * 16) + c['g'] + 'Training the model for ' +
                  c['b'] + '(%d parameters)' % net.count_trainable_parameters() + c['nc'])

            net.fit(x, y, batch_size=batch_size, validation_split=val_rate, epochs=epochs)

    net.save(net_name + '.mod')


def test_net(net, p, outputname):

    c = color_codes()
    options = parse_inputs()
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    p_name = p[0].rsplit('/')[-2]
    patient_path = '/'.join(p[0].rsplit('/')[:-1])
    outputname_path = os.path.join(patient_path, outputname + '.nii.gz')
    roiname = os.path.join(patient_path, outputname + '.roi.nii.gz')
    try:
        image = load_nii(outputname_path).get_data()
        load_nii(roiname)
    except IOError:
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Testing the network' + c['nc'])
        roi_nii = load_nii(p[0])
        roi = roi_nii.get_data().astype(dtype=np.bool)
        centers = get_mask_voxels(roi)
        test_samples = np.count_nonzero(roi)
        image = np.zeros_like(roi).astype(dtype=np.uint8)
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Creating the probability map ' + c['b'] + p_name + c['nc'] + c['g'] + ' - ' +
              c['b'] + outputname + c['nc'] + c['g'] + ' (%d samples)>' % test_samples + c['nc'])

        n_centers = len(centers)
        image_list = [load_norm_list(p)]
        is_roi = True
        roi = np.zeros_like(roi).astype(dtype=np.uint8)
        fcn_out = np.zeros_like(roi).astype(dtype=np.uint8)
        out = np.zeros_like(roi).astype(dtype=np.uint8)
        for i in range(0, n_centers, batch_size):
            print(
                '%f%% tested (step %d/%d)' % (100.0 * i / n_centers, (i / batch_size) + 1, -(-n_centers/batch_size)),
                end='\r'
            )
            sys.stdout.flush()
            centers_i = [centers[i:i + batch_size]]
            x = get_patches_list(image_list, centers_i, patch_size, True)
            x = np.concatenate(x).astype(dtype=np.float32)
            y_pr_pred = net.predict(x, batch_size=options['batch_size'])

            [x, y, z] = np.stack(centers_i[0], axis=1)

            out[x, y, z] = np.argmax(y_pr_pred[0], axis=1)
            y_pr_pred = y_pr_pred[1][:, y_pr_pred[1].shape[1]/2+1, :]
            y_pr_pred = np.squeeze(y_pr_pred)
            fcn_out[x, y, z] = np.argmax(y_pr_pred, axis=1)
            tumor = np.argmax(y_pr_pred, axis=1)

            # We store the ROI
            roi[x, y, z] = tumor.astype(dtype=np.bool)
            # We store the results
            y_pred = np.argmax(y_pr_pred, axis=1)
            image[x, y, z] = tumor if is_roi else y_pred

        print(' '.join([''] * 50), end='\r')
        sys.stdout.flush()

        # Post-processing (Basically keep the biggest connected region)
        image = get_biggest_region(image, is_roi)
        print(c['g'] + '                   -- Saving image ' + c['b'] + outputname_path + c['nc'])

        roi_nii.get_data()[:] = roi
        roi_nii.to_filename(roiname)
        roi_nii.get_data()[:] = get_biggest_region(fcn_out, is_roi)
        roi_nii.to_filename(os.path.join(patient_path, outputname + '.fcn.nii.gz'))

        roi_nii.get_data()[:] = get_biggest_region(out, is_roi)
        roi_nii.to_filename(os.path.join(patient_path, outputname + '.dense.nii.gz'))
        roi_nii.get_data()[:] = image
        roi_nii.to_filename(outputname_path)
    return image


def brats_main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    epochs = options['epochs']
    r_epochs = options['r_epochs']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    filters_list = n_filters if len(n_filters) > 1 else n_filters*conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks
    balanced = options['balanced']
    # Data loading parameters
    preload = options['preload']

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_train']
    filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    ub_s = '.ub' if not balanced else ''
    params_s = (ub_s, dfactor, patch_width, conv_s, filters_s, dense_size, epochs, r_epochs)
    sufix = '%s.D%d.p%d.c%s.n%s.d%d.e%d.E%d.' % params_s
    preload_s = ' (with ' + c['b'] + 'preloading' + c['nc'] + c['c'] + ')' if preload else ''

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting training' + preload_s + c['nc'])
    # N-fold cross validation main loop (we'll do 2 training iterations with testing for each patient)
    train_data, train_labels = get_names_from_path(options)

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + c['g'] +
          'Number of training images (%d=%d)' % (len(train_data), len(train_labels)) + c['nc'])
    #  Also, prepare the network

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Creating and compiling the model ' + c['nc'])
    input_shape = (train_data.shape[1],) + patch_size
    net = get_brats_net(input_shape, filters_list, kernel_size_list, dense_size, 2)
    net_name = os.path.join(path, 'brats2017-roi.tf' + sufix)
    train_net(net, net_name, 2)

    test_data, test_labels = get_names_from_path(options, False)
    dsc_results = list()
    for i, (p, gt_name) in enumerate(zip(test_data, test_labels)):
        p_name = p[0].rsplit('/')[-2]
        patient_path = '/'.join(p[0].rsplit('/')[:-1])
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + 'Case ' + c['c'] + c['b'] + p_name + c['nc'] +
              c['c'] + ' (%d/%d):' % (i + 1, len(test_data)) + c['nc'])
        image_name = os.path.join(patient_path, p_name + 'tensorflow.test')
        try:
            image = load_nii(image_name + '.nii.gz').get_data()
        except IOError:
            image = test_net(net, p, image_name)

        results = check_dsc(gt_name, image)
        dsc_string = c['g'] + '/'.join(['%f'] * len(results)) + c['nc']
        print(''.join([' '] * 14) + c['c'] + c['b'] + p_name + c['nc'] + ' DSC: ' + dsc_string)

        dsc_results.append(results)

    f_dsc = tuple([np.array([dsc[i] for dsc in dsc_results if len(dsc) > i]).mean() for i in range(3)])
    print('Final results DSC: (%f/%f/%f)' % f_dsc)


def check_dsc(gt_name, image):
    gt_nii = load_nii(gt_name)
    gt = np.copy(gt_nii.get_data()).astype(dtype=np.uint8)
    labels = np.unique(gt.flatten())
    return [dsc_seg(gt == l, image == l) for l in labels[1:]]


def main():
    # TODO: Basic Brats 2017 network like the #challenges2017 repository one
    c = color_codes()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    try:
        net = load_model('/home/mariano/Desktop/test.tf')
    except IOError:
        x = Input([784])
        x_image = Reshape([28, 28, 1])(x)
        x_conv1 = Conv(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(x_image)
        h_pool1 = MaxPool((2, 2), padding='same')(x_conv1)
        h_conv2 = Conv(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(h_pool1)
        h_pool2 = MaxPool((2, 2), padding='same')(h_conv2)
        h_fc1 = Dense(1024, activation='relu')(h_pool2)
        h_drop = Dropout(0.5)(h_fc1)
        y_conv = Dense(10)(h_drop)

        net = Model(x, y_conv, optimizer='adam', loss='categorical_cross_entropy', metrics='accuracy')

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] + c['b'] + 'Original (MNIST)' + c['nc'] +
         c['g'] + ' net ' + c['nc'] + c['b'] + '(%d parameters)' % net.count_trainable_parameters() + c['nc'])

    net.fit(
        mnist.train.images,
        mnist.train.labels,
        val_data=mnist.test.images,
        val_labels=mnist.test.labels,
        patience=10,
        epochs=200,
        batch_size=1024
    )

    save_model(net, '/home/mariano/Desktop/test.tf')

if __name__ == '__main__':
    # main()
    brats_main()

