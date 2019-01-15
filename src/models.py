"""
Defines networks.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer


def resnet50(x, nums, is_training=True, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    with tf.name_scope("Resnet", [x]):
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            net, end_points = resnet_v2.resnet_v2_50(       # shape=(N, 1, 1, 2048)
                x,
                num_classes=None,
                is_training=is_training,
                reuse=reuse,
                scope='resnet_v2_50')
            net = tf.squeeze(net, axis=[1, 2])              # shape=(N, 2048)
            net = slim.fully_connected(
                net, 
                num_outputs=nums,
                activation_fn=None, 
                trainable=is_training,
                reuse=reuse,
                scope='fc')
    variables = tf.contrib.framework.get_variables('resnet_v2_50')
    return net, variables
