# coding=utf-8
# Copyright @akikaaa.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import re
import numpy as np
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.estimator import util as estimator_util
from bert.run_classifier import *


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    # assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        name = name.split('model_body/')[-1]
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        #     mode=tf.estimator.ModeKeys.PREDICT,
        #     predictions=probabilities)
        output_spec = tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=probabilities
        )
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
        "ynt": YntProcessor,
    }

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()

    label_list = processor.get_labels()

    # tpu_cluster_resolver = None
    # if FLAGS.use_tpu and FLAGS.tpu_name:
    #     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    #         FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # run_config = tf.contrib.tpu.RunConfig(
    #     cluster=tpu_cluster_resolver,
    #     master=FLAGS.master,
    #     model_dir=FLAGS.output_dir,
    #     save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #     tpu_config=tf.contrib.tpu.TPUConfig(
    #         iterations_per_loop=FLAGS.iterations_per_loop,
    #         num_shards=FLAGS.num_tpu_cores,
    #         per_host_input_for_training=is_per_host))

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    )

    num_train_steps = None
    num_warmup_steps = None

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    # estimator = tf.contrib.tpu.TPUEstimator(
    #     use_tpu=FLAGS.use_tpu,
    #     model_fn=model_fn,
    #     config=run_config,
    #     train_batch_size=FLAGS.train_batch_size,
    #     eval_batch_size=FLAGS.eval_batch_size,
    #     predict_batch_size=FLAGS.predict_batch_size)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        model_dir=FLAGS.output_dir
    )

    feature_spec = {
        "input_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    # Build receiver function, and export.
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel('export', serving_input_receiver_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    tf.app.run()
