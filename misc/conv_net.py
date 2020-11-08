import string

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import Input


def Alex_Net(input: Input, regularization: string = None, dropout: float = .4) -> Sequential:
    model = Sequential()

    model.add(input)

    # layer 1
    model.add(Conv2D(filters=96, strides=4, kernel_size=11, activation='relu'))
    model.add(MaxPooling2D(strides=2, pool_size=3))

    # layer2
    model.add(Conv2D(filters=256, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=2, pool_size=3))

    # layer3-5
    for i in range(3):
        model.add(Conv2D(filters=384, kernel_size=3, activation='relu'))

    model.add(MaxPooling2D(pool_size=3, strides=2))

    model.add(Flatten())

    # FC Layers
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularization))
    model.add(Dropout(dropout))
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularization))

    # output layer
    model.add(Dense(1))

    return model
