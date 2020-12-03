import tensorflow as tf
    
from tensorflow import keras

bn_epsilon = 1e-5
bn_momentum = 0.1
lr_alpha = 0.2

def cbl(tf_input, filters, kernel_size, strides, is_training=False):
    # Need top left padding?
    out = keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(tf_input)
    out = keras.layers.BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum)(
        out, training=is_training)
    out = keras.layers.LeakyReLU(alpha=lr_alpha)(out)
    return out

def nn_base(input_img_size):
    tf_input = keras.Input(input_img_size)
    out = keras.layers.Conv2D(16, [3, 3], [1, 1], 'same')(tf_input)
    out = keras.layers.Conv2D(32, [3, 3], [2, 2], 'same')(out)
    out = keras.layers.Conv2D(64, [3, 3], [2, 2], 'same')(out)
    out = keras.layers.Conv2D(128, [3, 3], [2, 2], 'same')(out)
    # out = keras.layers.GlobalAvgPool2D()(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(1000)(out)
    out = keras.layers.Dense(30)(out)
    out = keras.activations.sigmoid(out)
    return keras.Model(inputs=tf_input, outputs=out)

def nn_cbl(input_img_size, is_training=False):
    tf_input = keras.Input(input_img_size)
    out = cbl(tf_input, 16, [3, 3], [1, 1], is_training)
    out = cbl(out, 32, [3, 3], [2, 2], is_training)
    out = cbl(out, 64, [3, 3], [2, 2], is_training)
    out = cbl(out, 128, [3, 3], [2, 2], is_training)
    out = keras.layers.GlobalAvgPool2D()(out)
    out = keras.layers.Dense(30)(out)
    out = keras.activations.sigmoid(out)
    return keras.Model(inputs=tf_input, outputs=out)

def nn_new(input_img_size, is_training=False):
    tf_input = keras.Input(input_img_size)
    out = keras.layers.Conv2D(256, [3, 3], 2, activation='relu')(tf_input)
    out = keras.layers.Conv2D(256, [3, 3], 2, activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Conv2D(128, [3, 3], 1, activation='relu')(out)
    out = keras.layers.Conv2D(128, [3, 3], 1, activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Conv2D(128, [3, 3], 1, activation='relu')(out)
    out = keras.layers.Conv2D(128, [3, 3], 1, activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Conv2D(64, [3, 3], 1, activation='relu')(out)
    out = keras.layers.Conv2D(64, [3, 3], 1, activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Conv2D(32, [3, 3], 1, activation='relu')(out)
    out = keras.layers.Conv2D(32, [3, 3], 1, activation='relu')(out)
    out = keras.layers.BatchNormalization()(out)

    out = keras.layers.Conv2D(30, [3, 3], 1, activation='relu')(out)
    out = keras.layers.Conv2D(30, [3, 3], 1, activation='relu')(out)
    out = keras.layers.Conv2D(30, [3, 3], 1)(out)
    out = keras.layers.Flatten()(out)

    return keras.Model(inputs=tf_input, outputs=out)
