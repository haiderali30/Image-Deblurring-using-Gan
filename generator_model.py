import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    
    conv1 = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), padding='same')(pool1)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.LeakyReLU(alpha=0.2)(conv3)

    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    conv4 = layers.Conv2D(64, (3, 3), padding='same')(up1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.LeakyReLU(alpha=0.2)(conv4)

    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    conv5 = layers.Conv2D(32, (3, 3), padding='same')(up2)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.LeakyReLU(alpha=0.2)(conv5)

    outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(conv5)

    return models.Model(inputs, outputs)
