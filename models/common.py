import tensorflow as tf

from tensorflow.keras.layers import ZeroPadding2D, Conv2D, LeakyReLU
from tensorflow.keras.regularizers import l2


class BatchNormalization(tf.keras.layers.BatchNormalization):

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super(BatchNormalization, self).call(x, training)


def convolutional(input_layer, filters_shape, down_sample=False, activate=True, bn=True, activate_fn='leaky'):
    if down_sample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'
    filters = filters_shape[-1]
    kernel_size = filters_shape[0]
    use_bias = not bn
    kernel_regularizer = l2(0.0005)
    kernel_init = tf.random_normal_initializer(stddev=0.01)
    bias_init = tf.constant_initializer(0.)
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                  kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_init, bias_initializer=bias_init)(
        input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate:
        if activate_fn == 'leaky':
            conv = LeakyReLU(alpha=0.1)(conv)
        elif activate_fn == 'mish':
            conv = mish(conv)
    return conv


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_fn='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_fn=activate_fn)
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2), activate_fn=activate_fn)
    residual_output = short_cut + conv
    return residual_output


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def up_sample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')
