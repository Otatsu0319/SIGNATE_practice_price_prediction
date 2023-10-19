# resnet modules
import tensorflow as tf

class clip_value(tf.keras.layers.Layer):

    def __init__(self, clip_value_min = -6000.0, clip_value_max = 6000.0, **kwargs):
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max
        super(clip_value, self).__init__(**kwargs)

    def call(self, inputs): 
        return tf.clip_by_value(inputs, self.clip_value_min, self.clip_value_max)


# Code: https://qiita.com/koshian2/items/6742c469e9775d672072
def se_block(inputs, channels, r=8):
    # Squeeze
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    # Excitation
    x = tf.keras.layers.Dense(channels//r, activation="relu")(x)
    x = tf.keras.layers.Dense(channels, activation="sigmoid")(x)
    return tf.keras.layers.Multiply()([inputs, x])

def conv2D_unit(inputs, filters, kernel_size, use_dropout=False):
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    if use_dropout:
        x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same")(x)
    return outputs

def conv2D_L_unit(inputs, filters, kernel_size, strides):
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    outputs = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer="he_normal", padding="same")(x)
    return outputs

def conv2D_R_unit(inputs, filters, kernel_size, strides):
    x = tf.keras.layers.BatchNormalization()(inputs)
    outputs = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, kernel_initializer="he_normal", padding="same")(x)
    return outputs

def first_res_block(inputs):
    x = tf.keras.layers.Conv2D(64, kernel_size=7,strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    first_output = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
    return first_output

def Conv_res_block(inputs, filters, use_bneck):
    if use_bneck:
        x_L = conv2D_L_unit(inputs, filters, kernel_size=1, strides=2)
        x_L = conv2D_L_unit(x_L, filters, kernel_size=3, strides=1)
        x_L = conv2D_L_unit(x_L, filters*4, kernel_size=1, strides=1)

        x_R = conv2D_R_unit(inputs, filters*4, kernel_size=1, strides=2)

    else:
        x_L = conv2D_L_unit(inputs, filters, kernel_size=1, strides=2)
        x_L = conv2D_L_unit(x_L, filters, kernel_size=3, strides=1)

        x_R = conv2D_R_unit(inputs, filters, kernel_size=1, strides=2)

    stage = tf.keras.layers.add([x_L, x_R])
    return stage

def Identity_res_block(inputs, filters, use_bneck):
    if use_bneck:
        x = conv2D_unit(inputs, filters, kernel_size=1)
        x = conv2D_unit(x, filters, kernel_size=3)
        x = conv2D_unit(x, filters*4, kernel_size=1, use_dropout=True)
    
    else:
        x = conv2D_unit(inputs, filters, kernel_size=3)
        x = conv2D_unit(x, filters, kernel_size=3, use_dropout=True)
    outputs = tf.keras.layers.add([x, inputs])
    return outputs


def first_res_D_block(inputs):
    x = tf.keras.layers.Conv2D(64, kernel_size=3,strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3,strides=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3,strides=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    first_output = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

    return first_output

def Conv_res_D_block(inputs, filters, use_bneck):

    if use_bneck:
        x_L = conv2D_L_unit(inputs, filters, kernel_size=1, strides=1)
        x_L = conv2D_L_unit(x_L, filters, kernel_size=3, strides=2)

        #x_L = se_block(x_L, filters)

        x_L = conv2D_L_unit(x_L, filters*4, kernel_size=1, strides=1)


        x_R = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(inputs)
        x_R = conv2D_R_unit(x_R, filters*4, kernel_size=1, strides=1)

    else:
        x_L = conv2D_L_unit(inputs, filters, kernel_size=1, strides=2)

        x_L = se_block(x_L, filters)

        x_L = conv2D_L_unit(x_L, filters, kernel_size=3, strides=1)


        x_R = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(inputs)
        x_R = conv2D_R_unit(x_R, filters, kernel_size=1, strides=1)

    stage = tf.keras.layers.add([x_L, x_R])
    return stage

def Identity_res_D_block(inputs, filters, use_bneck):
    if use_bneck:
        x = conv2D_L_unit(inputs, filters, kernel_size=1,  strides=(1, 1))
        x = conv2D_L_unit(x, filters, kernel_size=3,  strides=(1, 1))

        #x = se_block(x, filters)

        x = conv2D_L_unit(x, filters*4, kernel_size=1,  strides=(1, 1))

    else:
        x = conv2D_L_unit(inputs, filters, kernel_size=3,  strides=(1, 1))

        x = se_block(x, filters)

        x = conv2D_L_unit(x, filters, kernel_size=3,  strides=(1, 1))

    outputs = tf.keras.layers.add([x, inputs])
    return outputs
