import tensorflow as tf
import models.common as common


def darknet53(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 32))
    input_data = common.convolutional(input_data, (3, 3, 32, 64), down_sample=True)

    for i in range(1):
        input_data = common.residual_block(input_data, 64, 32, 64)

    input_data = common.convolutional(input_data, (3, 3, 64, 128), down_sample=True)

    for i in range(2):
        input_data = common.residual_block(input_data, 128, 64, 128)

    input_data = common.convolutional(input_data, (3, 3, 128, 256), down_sample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), down_sample=True)

    for i in range(8):
        input_data = common.residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), down_sample=True)

    for i in range(4):
        input_data = common.residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data


def cspdarknet53(input_data):
    input_data = common.convolutional(input_data, (3, 3, 3, 32), activate_fn="mish")
    input_data = common.convolutional(input_data, (3, 3, 32, 64), down_sample=True, activate_fn="mish")

    route = input_data
    route = common.convolutional(route, (1, 1, 64, 64), activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_fn="mish")
    for i in range(1):
        input_data = common.residual_block(input_data, 64, 32, 64, activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_fn="mish")

    input_data = tf.concat([input_data, route], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_fn="mish")
    input_data = common.convolutional(input_data, (3, 3, 64, 128), down_sample=True, activate_fn="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 128, 64), activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 64), activate_fn="mish")
    for i in range(2):
        input_data = common.residual_block(input_data, 64, 64, 64, activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 64, 64), activate_fn="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_fn="mish")
    input_data = common.convolutional(input_data, (3, 3, 128, 256), down_sample=True, activate_fn="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 256, 128), activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 128), activate_fn="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 128, 128, 128, activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 128, 128), activate_fn="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_fn="mish")
    route_1 = input_data
    input_data = common.convolutional(input_data, (3, 3, 256, 512), down_sample=True, activate_fn="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 512, 256), activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 256), activate_fn="mish")
    for i in range(8):
        input_data = common.residual_block(input_data, 256, 256, 256, activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 256, 256), activate_fn="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_fn="mish")
    route_2 = input_data
    input_data = common.convolutional(input_data, (3, 3, 512, 1024), down_sample=True, activate_fn="mish")
    route = input_data
    route = common.convolutional(route, (1, 1, 1024, 512), activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512), activate_fn="mish")
    for i in range(4):
        input_data = common.residual_block(input_data, 512, 512, 512, activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 512, 512), activate_fn="mish")
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = common.convolutional(input_data, (1, 1, 1024, 1024), activate_fn="mish")
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1),
                            tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1),
                            tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = common.convolutional(input_data, (1, 1, 2048, 512))
    input_data = common.convolutional(input_data, (3, 3, 512, 1024))
    input_data = common.convolutional(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data
