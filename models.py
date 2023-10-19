import tensorflow as tf
import modules


def build_model_regressor():
    inputs = tf.keras.layers.Input(shape=(68, 1))

    # first block
    x = tf.keras.layers.Dense(256)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Reshape((68, 32, 32))(x)

    block_output = modules.first_res_block(x)

    # res stack
    stacks = [3, 4, 6, 3]  # [2, 2, 2, 2]
    bneck_bool = True
    for i, stack in enumerate(stacks):
        block_output = modules.Conv_res_block(inputs=block_output, filters=64 * (2 ** (i)), use_bneck=bneck_bool)
        for _ in range(stack):
            block_output = modules.Identity_res_block(block_output, 64 * (2 ** (i)), bneck_bool)

    x = tf.keras.layers.GlobalAveragePooling2D()(block_output)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Activation('linear', dtype=tf.float32)(x)

    model = tf.keras.Model(inputs, outputs, name="resnet")
    return model


def build_model_regressor_v2():
    inputs = tf.keras.layers.Input(shape=(68, 1))

    x = inputs
    for unit in [128, 256, 512, 1024]:
        x_ = tf.keras.layers.Dense(unit)(x)
        x_ = tf.keras.layers.BatchNormalization()(x_)
        x_ = tf.keras.layers.LeakyReLU(alpha=0.01)(x_)
        x = tf.keras.layers.Dropout(0.3)(x_)

    x = tf.keras.layers.Dense(1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Activation('linear', dtype=tf.float32)(x)

    model = tf.keras.Model(inputs, outputs, name="resnet")
    return model
