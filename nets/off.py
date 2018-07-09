from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, batch_norm, conv2d
from nets import nets_factory

# tf.add_check_numerics_ops()

# slim = tf.contrib.slim

_NUM_CHANELS = 128

sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])


def sobel(feature):
    with tf.variable_scope('sobel'):
        channels = tf.unstack(feature, axis=3)
        fx = []
        fy = []

        for channel in channels:
            channel = tf.expand_dims(channel, axis=3)
            filtered_x = tf.nn.conv2d(channel, sobel_x_filter, [1, 1, 1, 1], padding='SAME')
            filtered_y = tf.nn.conv2d(channel, sobel_y_filter, [1, 1, 1, 1], padding='SAME')
            fx.append(filtered_x)
            fy.append(filtered_y)

        return tf.concat(fx, axis=3), tf.concat(fy, axis=3)


def _padding(tensor, out_size):
    t_width = tensor.get_shape()[1]
    delta = tf.subtract(out_size, t_width)
    pad_left = tf.floor_div(delta, 2)
    pad_right = delta - pad_left
    return tf.pad(
        tensor,
        [
            [0, 0],
            [pad_left, pad_right],
            [pad_left, pad_right],
            [0, 0]
        ],
        'CONSTANT'
    )


def padding_and_concat(list_feature, out_size):
    padded_list = []
    for item in list_feature:
        padded = tf.cond(tf.equal(out_size, item.get_shape()[1]),
                         lambda: item,
                         lambda: _padding(item, out_size))
        shape = item.get_shape()
        padded.set_shape([shape[0], out_size, out_size, shape[3]])
        padded_list.append(padded)

    return tf.concat(padded_list, axis=3)


def off_unit(feature_t0, feature_t1, lower_unit):
    with tf.variable_scope('off_unit', values=[feature_t0, feature_t1]):
        # feature_t0 = batch_norm(feature_t0)
        # feature_t0 = tf.nn.relu(feature_t0)
        feature_t0 = conv2d(feature_t0, _NUM_CHANELS, 1, padding='SAME',
                            # weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                            # biases_initializer=tf.zeros_initializer,
                            # weights_regularizer=l2_regularizer(1e-3),
                            # biases_regularizer=l2_regularizer(0.0001),
                            # normalizer_fn=batch_norm,
                            scope='conv1x1_t0')

        # feature_t1 = batch_norm(feature_t1)
        # feature_t1 = tf.nn.relu(feature_t1)
        feature_t1 = conv2d(feature_t1, _NUM_CHANELS, 1, padding='SAME',
                            # weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                            # biases_initializer=tf.zeros_initializer,
                            # weights_regularizer=l2_regularizer(1e-3),
                            # biases_regularizer=l2_regularizer(0.0001),
                            # normalizer_fn=batch_norm,
                            scope='conv1x1_t1')

        ft = tf.subtract(feature_t0, feature_t1)
        fx, fy = sobel(feature_t0)

        return tf.concat(
            [
                fx,
                fy,
                ft,
                lower_unit
            ],
            axis=3
        )


def off_unit_first(feature_t0, feature_t1):
    with tf.variable_scope('off_unit_first', values=[feature_t0, feature_t1]):
        # feature_t0 = batch_norm(feature_t0)
        # feature_t0 = tf.nn.relu(feature_t0)
        feature_t0 = conv2d(feature_t0, _NUM_CHANELS, 1, padding='SAME',
                            # weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                            # biases_initializer=tf.zeros_initializer,
                            # weights_regularizer=l2_regularizer(1e-3),
                            # biases_regularizer=l2_regularizer(0.0001),
                            # normalizer_fn=batch_norm,
                            scope='conv1x1_t0')

        # feature_t1 = batch_norm(feature_t1)
        # feature_t1 = tf.nn.relu(feature_t1)
        feature_t1 = conv2d(feature_t1, _NUM_CHANELS, 1, padding='SAME',
                            # weights_initializer=tf.truncated_normal_initializer(0, 0.01),
                            # biases_initializer=tf.zeros_initializer,
                            # weights_regularizer=l2_regularizer(1e-3),
                            # biases_regularizer=l2_regularizer(0.0001),
                            # normalizer_fn=batch_norm,
                            scope='conv1x1_t1')

        ft = tf.subtract(feature_t0, feature_t1)
        fx, fy = sobel(feature_t0)

        return tf.concat(
            [
                fx,
                fy,
                ft
            ],
            axis=3
        )


def off_sub_network(list_feature_k0, list_feature_k1,
                    list_feature_k20, list_feature_k21,
                    list_feature_k40, list_feature_k41, num_classes=11, is_training=True):
    '''
    :param list_feature_k0: list of feature with maximum size (k size) of segment t
    :param list_feature_k1: list of feature with maximum size (k size) of segment t + delta_t
    :param list_feature_k20: list of feature with k/2 size of segment t
    :param list_feature_k21: list of feature with k/2 size of segment t + delta_t
    :param list_feature_k40: list of feature with k/4 size of segment t
    :param list_feature_k41: list of feature with k/4 size of segment t + delta_t
    :param num_classes: number classes
    :return: logits, endpoints
    '''

    resnet_v2_20 = nets_factory.get_network_fn(
        'resnet_v2_26',
        num_classes=num_classes,
        weight_decay=0.001,
        is_training=is_training)

    endpoints = {}

    with tf.variable_scope('OFFSubNetwork'):
        with tf.variable_scope('Tier1'):
            feature_k0 = padding_and_concat(list_feature_k0, 111)
            feature_k1 = padding_and_concat(list_feature_k1, 111)

            net = off_unit_first(feature_k0, feature_k1)
            logits, tier1_enpoints = resnet_v2_20(net)
            endpoints['tier1'] = tier1_enpoints['OFF/OFFSubNetwork/Tier1/resnet_v2_26/block4']
            endpoints['logits_tier1'] = logits
        with tf.variable_scope('Tier2'):
            feature_k20 = padding_and_concat(list_feature_k20, 56)
            feature_k21 = padding_and_concat(list_feature_k21, 56)

            net = off_unit(feature_k20, feature_k21, lower_unit=endpoints['tier1'])
            logits, tier2_endpoint = resnet_v2_20(net)
            endpoints['tier2'] = tier2_endpoint['OFF/OFFSubNetwork/Tier2/resnet_v2_26/block4']
            endpoints['logits_tier2'] = logits
        with tf.variable_scope('Tier3'):
            feature_k40 = padding_and_concat(list_feature_k40, 28)
            feature_k41 = padding_and_concat(list_feature_k41, 28)

            net = off_unit(feature_k40, feature_k41, lower_unit=endpoints['tier2'])
            logits, tier3_endpoints = resnet_v2_20(net)
            endpoints['logits_tier3'] = logits
            endpoints['predictions'] = tier3_endpoints['predictions']

            return logits, endpoints
