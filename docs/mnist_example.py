"""
Trains a simple neural network on the MNIST dataset.

Gets to 97.54% test accuracy after 10 epochs.
22 seconds per epoch on a NVIDIA Geforce 940MX.
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

# internal
from clana.io import write_predictions, write_gt


def main():
    batch_size = 128
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Write gt for CLANA
    write_gt(dict(enumerate(y_train)), 'gt-train.csv')
    write_gt(dict(enumerate(y_test)), 'gt-test.csv')

    x_train, y_train = preprocess(x_train, y_train, img_rows, img_cols, num_classes)
    x_test, y_test = preprocess(x_test, y_test, img_rows, img_cols, num_classes)
    input_shape = get_shape(img_rows, img_cols)
    model = create_model(input_shape, num_classes)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_train)

    # Write gt for CLANA
    y_train_pred_a = np.argmax(y_train_pred, axis=1)
    y_test_pred_a = np.argmax(y_test_pred, axis=1)
    write_predictions(dict(enumerate(y_train_pred_a)), 'train-pred.csv')
    write_predictions(dict(enumerate(y_test_pred_a)), 'test-pred.csv')

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def get_shape(img_rows, img_cols):
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    return input_shape


def preprocess(features, targets, img_rows, img_cols, num_classes):
    if K.image_data_format() == 'channels_first':
        features = features.reshape(features.shape[0], 1, img_rows, img_cols)
    else:
        features = features.reshape(features.shape[0], img_rows, img_cols, 1)
    features = features.astype('float32')
    features /= 255
    print('x shape:', features.shape)
    print('{} samples'.format(features.shape[0]))

    # convert class vectors to binary class matrices
    targets = keras.utils.to_categorical(targets, num_classes)
    return features, targets


def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
