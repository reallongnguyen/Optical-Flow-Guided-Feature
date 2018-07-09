# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import ucf11
from nets import nets_factory
from preprocessing import preprocessing_factory
from nets.off_v2 import off

# from tensorflow.python.ops import variable_scope
# from tensorflow.python.ops import array_ops
# from tensorflow.python.framework import ops
# from tensorflow.python.framework import dtypes
# from tensorflow.python.ops import confusion_matrix
# from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import state_ops

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 16, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'mode', 'fast', 'Eval che do nhanh hoac che do day du.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'results/ucf11-off/train',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', 'results/ucf11-off/test', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'ucf11', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', 'data/UCF11-tfrecord', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'resnet_model_name', 'resnet_v2_26', 'The name of the resnet model to evaluate')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'off', 'The name of the preprocessing to use. If left '
                                 'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def get_basic_feature(endpoints):
    f_k = [
        endpoints['Conv2d_1a_7x7'],
    ]

    # Feature with size k/2
    f_k2 = [
        endpoints['MaxPool_2a_3x3'],
        endpoints['Conv2d_2b_1x1'],
        endpoints['Conv2d_2c_3x3'],
    ]

    # Feature with size k/4
    f_k4 = [
        endpoints['MaxPool_3a_3x3'],
        endpoints['Mixed_3b'],
        endpoints['Mixed_3c'],
    ]
    return f_k, f_k2, f_k4


# def metric_variable(shape, dtype, validate_shape=True, name=None):
#     """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections."""
#
#     return variable_scope.variable(
#         lambda: array_ops.zeros(shape, dtype),
#         trainable=False,
#         collections=[
#             ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
#         ],
#         validate_shape=validate_shape,
#         name=name)


# def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
#     """Calculate a streaming confusion matrix.
#     Calculates a confusion matrix. For estimation over a stream of data,
#     the function creates an  `update_op` operation.
#     Args:
#       labels: A `Tensor` of ground truth labels with shape [batch size] and of
#         type `int32` or `int64`. The tensor will be flattened if its rank > 1.
#       predictions: A `Tensor` of prediction results for semantic labels, whose
#         shape is [batch size] and type `int32` or `int64`. The tensor will be
#         flattened if its rank > 1.
#       num_classes: The possible number of labels the prediction task can
#         have. This value must be provided, since a confusion matrix of
#         dimension = [num_classes, num_classes] will be allocated.
#       weights: Optional `Tensor` whose rank is either 0, or the same rank as
#         `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
#         be either `1`, or the same as the corresponding `labels` dimension).
#     Returns:
#       total_cm: A `Tensor` representing the confusion matrix.
#       update_op: An operation that increments the confusion matrix.
#     """
#     # Local variable to accumulate the predictions in the confusion matrix.
#     total_cm = metric_variable(
#         [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')
#
#     # Cast the type to int64 required by confusion_matrix_ops.
#     predictions = math_ops.to_int64(predictions)
#     labels = math_ops.to_int64(labels)
#     num_classes = math_ops.to_int64(num_classes)
#
#     # Flatten the input if its rank > 1.
#     if predictions.get_shape().ndims > 1:
#         predictions = array_ops.reshape(predictions, [-1])
#
#     if labels.get_shape().ndims > 1:
#         labels = array_ops.reshape(labels, [-1])
#
#     if (weights is not None) and (weights.get_shape().ndims > 1):
#         weights = array_ops.reshape(weights, [-1])
#
#     # Accumulate the prediction to current confusion matrix.
#     current_cm = confusion_matrix.confusion_matrix(
#         labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
#     update_op = state_ops.assign_add(total_cm, current_cm)
#
#     return total_cm, update_op


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    # with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = ucf11.get_split(FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    input, label = ucf11.build_data(dataset, is_training=False)
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    input = image_preprocessing_fn(input, eval_image_size, eval_image_size)

    inputs, labels = tf.train.batch(
        [input, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    images = tf.unstack(inputs, axis=1)

    def off_rgb(image1, image2, name):
        logits1, endpoints1 = network_fn(image1)
        logits2, endpoints2 = network_fn(image2)

        # Feature with size maximum k
        f_k_1, f_k2_1, f_k4_1 = get_basic_feature(endpoints1)
        f_k_2, f_k2_2, f_k4_2 = get_basic_feature(endpoints2)

        logits_off, end_point_off = off(
            f_k_1, f_k_2,
            f_k2_1, f_k2_2,
            f_k4_1, f_k4_2,
            num_classes=dataset.num_classes - FLAGS.labels_offset,
            resnet_model_name=FLAGS.resnet_model_name,
            resnet_weight_decay=0.0,
            is_training=False
        )

        logits_gen = tf.reduce_mean(tf.stack([
            logits1,
            logits2
        ], axis=2), axis=2)

        logits = tf.multiply(logits_gen, logits_off, name='logits' + name)

        return logits

    logits_arr = []

    max_range = 1 if FLAGS.mode == 'fast' else 10

    for i in range(0, max_range, 1):
        logits = off_rgb(images[i], images[i + 1], str(i))
        logits_arr.append(logits)

    logits = tf.reduce_mean(tf.stack(logits_arr, axis=2), axis=2)

    if FLAGS.moving_average_decay:
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, tf_global_step)
        variables_to_restore = variable_averages.variables_to_restore(
            slim.get_model_variables())
        variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
        # 'Confusion': _streaming_confusion_matrix(labels, predictions, dataset.num_classes)
        # get_streaming_metrics(predictions, labels, dataset.num_classes),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
        # if name != 'Confusion':
        summary_name = 'eval/%s' % name
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        # else:
        #     summary_name = 'eval/%s' % name
        #     c_image = tf.reshape(tf.cast(value, tf.float32), [1, 11, 11, 1])
        #     tf.summary.image('confusion_image', c_image)
        #     op = tf.summary.tensor_summary(summary_name, c_image, collections=[])
        #     op = tf.Print(op, [value], summary_name, summarize=121)
        #     tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
        num_batches = FLAGS.max_num_batches
    else:
        # This ensures that we make a single pass over all of the data.
        num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    # update_ops = []
    # update_ops.extend(list(names_to_updates.values()))
    # update_ops.extend(confusion_op)
    # update_op = tf.group(*update_ops)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)

    # print(confusion.eval())


if __name__ == '__main__':
    tf.app.run()
