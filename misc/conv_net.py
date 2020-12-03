import string

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras import Input


def alex_net(input: Input, regularization: string = None, dropout: float = .4) -> Sequential:
    """
    modified alex net to find relationship between feature vectors and chem prop
    :param input: Keras Input Dimensions
    :param regularization:
    :param dropout:
    :return:
    """
    model = Sequential()

    model.add(input)

    # layer 1
    model.add(Conv2D(filters=96, strides=4, kernel_size=11, activation='relu'))
    model.add(MaxPooling2D(strides=2, pool_size=3))

    # layer2
    model.add(Conv2D(filters=256, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=2, pool_size=3))

    # layer3-5
    model.add(Conv2D(filters=384, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=3, strides=2))

    model.add(Flatten())

    # FC Layers
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularization))
    model.add(Dropout(dropout))
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularization))

    # output layer
    model.add(Dense(1))

    return model


def light_alex_net(input: Input, regularization: string = None, dropout: float = .4):

    model = Sequential()
    model.add(input)

    #layer 2
    model.add(Conv2D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=2, pool_size=3, padding='same'))

    #layer 3
    model.add(Conv2D(filters=256, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=2, pool_size=3, padding='same'))

    model.add(Flatten())

    #FC Layers
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularization))
    model.add(Dropout(dropout))
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularization))

    #output layer
    model.add(Dense(1))

    return model

def light_alex_net_v2(input: Input, regularization: string = None, dropout: float = .4):

    model = Sequential()
    model.add(input)

    #layer 2
    model.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=2, pool_size=3, padding='same'))

    #layer 3
    model.add(Conv2D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(strides=2, pool_size=3, padding='same'))

    model.add(Flatten())

    #FC Layers
    model.add(Dense(2048, activation='relu', kernel_regularizer=regularization))
    model.add(Dropout(dropout))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularization))

    #output layer
    model.add(Dense(1))

    return model