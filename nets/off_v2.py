from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer, batch_norm, conv2d
from nets import nets_factory

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


def _concat(list_feature):
    return tf.concat(list_feature, axis=3)


def off_unit(feature_t0, feature_t1, lower_unit):
    with tf.variable_scope('off_unit', values=[feature_t0, feature_t1]):
        feature_t0 = conv2d(feature_t0, _NUM_CHANELS, 1, padding='SAME',
                            scope='conv1x1_t0')

        feature_t1 = conv2d(feature_t1, _NUM_CHANELS, 1, padding='SAME',
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
        feature_t0 = conv2d(feature_t0, _NUM_CHANELS, 1, padding='SAME',
                            scope='conv1x1_t0')

        feature_t1 = conv2d(feature_t1, _NUM_CHANELS, 1, padding='SAME',
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


def off(list_feature_k0, list_feature_k1,
                    list_feature_k20, list_feature_k21,
                    list_feature_k40, list_feature_k41,
                    num_classes=11,
                    is_training=True,
                    resnet_model_name='resnet_v2_26',
                    resnet_weight_decay=0.004):
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

    resnet = nets_factory.get_network_fn(
        resnet_model_name,
        num_classes=num_classes,
        weight_decay=resnet_weight_decay,
        is_training=is_training)

    endpoints = {}

    with tf.variable_scope('OFF', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('Tier1'):
            feature_k0 = _concat(list_feature_k0)
            feature_k1 = _concat(list_feature_k1)

            net = off_unit_first(feature_k0, feature_k1)
            logits, tier1_enpoints = resnet(net)
            endpoints['tier1'] = tier1_enpoints['OFF/Tier1/%s/block4' % resnet_model_name]
            endpoints['logits_tier1'] = logits
        with tf.variable_scope('Tier2'):
            feature_k20 = _concat(list_feature_k20)
            feature_k21 = _concat(list_feature_k21)

            net = off_unit(feature_k20, feature_k21, lower_unit=endpoints['tier1'])
            logits, tier2_endpoint = resnet(net)
            endpoints['tier2'] = tier2_endpoint['OFF/Tier2/%s/block4' % resnet_model_name]
            endpoints['logits_tier2'] = logits
        with tf.variable_scope('Tier3'):
            feature_k40 = _concat(list_feature_k40)
            feature_k41 = _concat(list_feature_k41)

            net = off_unit(feature_k40, feature_k41, lower_unit=endpoints['tier2'])
            logits, tier3_endpoints = resnet(net)
            endpoints['logits_tier3'] = logits
            endpoints['predictions'] = tier3_endpoints['predictions']

            return logits, endpoints
