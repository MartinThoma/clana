"""
Trains a simple neural network on the MNIST dataset.

Gets to 97.54% test accuracy after 10 epochs.
22 seconds per epoch on a NVIDIA Geforce 940MX.
"""

# Core Library
from typing import Any, Tuple

# Third party
import keras
import numpy as np
import numpy.typing as npt
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

# First party
from clana.io import write_gt, write_predictions


def main() -> None:
    batch_size = 128
    num_classes = 10
    epochs = 1

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Write gt for CLANA
    write_gt(dict(enumerate(y_train)), "gt-train.csv")  # type: ignore
    write_gt(dict(enumerate(y_test)), "gt-test.csv")  # type: ignore

    x_train, y_train = preprocess(x_train, y_train, img_rows, img_cols, num_classes)
    x_test, y_test = preprocess(x_test, y_test, img_rows, img_cols, num_classes)
    input_shape = get_shape(img_rows, img_cols)
    model = create_model(input_shape, num_classes)

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_train)

    # Write gt for CLANA
    y_train_pred_a = np.argmax(y_train_pred, axis=1)
    y_test_pred_a = np.argmax(y_test_pred, axis=1)
    write_predictions(dict(enumerate(y_train_pred_a)), "train-pred.csv")  # type: ignore
    write_predictions(dict(enumerate(y_test_pred_a)), "test-pred.csv")  # type: ignore

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


def get_shape(img_rows: int, img_cols: int) -> Tuple[int, int, int]:
    if K.image_data_format() == "channels_first":
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    return input_shape


def preprocess(
    features: npt.NDArray,
    targets: npt.NDArray,
    img_rows: int,
    img_cols: int,
    num_classes: int,
) -> Tuple[Any, Any]:
    if K.image_data_format() == "channels_first":
        features = features.reshape(features.shape[0], 1, img_rows, img_cols)
    else:
        features = features.reshape(features.shape[0], img_rows, img_cols, 1)
    features = features.astype("float32")
    features /= 255
    print("x shape:", features.shape)
    print(f"{features.shape[0]} samples")

    # convert class vectors to binary class matrices
    targets = keras.utils.to_categorical(targets, num_classes)
    return features, targets


def create_model(input_shape: Tuple[int, int, int], num_classes: int) -> Any:
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    main()
