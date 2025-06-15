import tensorflow as tf
from tensorflow.keras import layers, models

def build_discriminator(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
    conv1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), padding='same')(conv1)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)
    conv2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), padding='same')(conv2)
    conv3 = layers.LeakyReLU(alpha=0.2)(conv3)
    conv3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), padding='same')(conv3)
    conv4 = layers.LeakyReLU(alpha=0.2)(conv4)
    conv4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    flatten = layers.Flatten()(conv4)
    dense1 = layers.Dense(1024)(flatten)
    dense1 = layers.LeakyReLU(alpha=0.2)(dense1)

    outputs = layers.Dense(1, activation='sigmoid')(dense1)

    return models.Model(inputs, outputs)
