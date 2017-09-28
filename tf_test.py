from __future__ import print_function
from time import strftime
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from models import Model, save_model, load_model
from layers import Input, Reshape, Conv, MaxPool, Dense, Dropout
from utils import color_codes


def main():
    # TODO: Basic Brats 2017 network like the #challenges2017 repository one
    c = color_codes()
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    try:
        net = load_model('/home/mariano/Desktop/test.tf')
    except IOError:
        x = Input([784])
        x_image = Reshape([-1, 28, 28, 1])(x)
        x_conv1 = Conv(filters=32, kernel_size=(5, 5), activation='relu', padding='SAME')(x_image)
        h_pool1 = MaxPool((2, 2), padding='SAME')(x_conv1)
        h_conv2 = Conv(filters=64, kernel_size=(5, 5), activation='relu', padding='SAME')(h_pool1)
        h_pool2 = MaxPool((2, 2), padding='SAME')(h_conv2)
        h_fc1 = Dense(1024, activation='relu')(h_pool2)
        h_drop = Dropout(0.5)(h_fc1)
        y_conv = Dense(10)(h_drop)

        net = Model(x, y_conv, optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

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

    print(np.argmax(net.predict(mnist.test.images, batch_size=1024), axis=1))
    print(np.argmax(mnist.test.labels, axis=1))


if __name__ == '__main__':
    main()

