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


class MyProcessor(DataProcessor):

    def get_test_examples(self, data_dir):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_pred_examples(self, data_dir):
        """See base class."""
        # lines = self._read_tsv(os.path.join(data_dir, "pred.tsv"))
        # print('num of lines: %s '%len(lines))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "pred.tsv")), "pred")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets. each line is label+\t+text_a+\t+text_b """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def _has_dataset_or_queue_runner(maybe_tensor):
    """Returns True if TF dataset or QueueRunner has been used."""
    # Check TF dataset first. Here, we use a simple algorithm to check the top
    # level Tensors only, which should be sufficient for most users.
    tensors = [x for x in nest.flatten(maybe_tensor) if isinstance(x, ops.Tensor)]
    if any([t.op.type == 'IteratorGetNext' for t in tensors]):
        return True


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


class Classifier:
    def __init__(self):
        self.processor = MyProcessor()
        # self.estimator = self.create_estimator()
        # self.sess = tf.train.MonitoredSession()
        self.sess = tf.Session()
        self.initialized = False

    def model_fn(self, features, bert_config, num_labels, init_checkpoint):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        with tf.variable_scope('model_body', reuse=tf.AUTO_REUSE):
            (total_loss, per_example_loss, logits, probabilities) = create_model(
                bert_config, False, input_ids, input_mask, segment_ids, label_ids,
                num_labels, False)

        tvars = tf.trainable_variables()

        (assignment_map, initialized_variable_names
         ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return assignment_map, probabilities, initialized_variable_names

    def predict_online(self):
        # tf.reset_default_graph()
        predict_examples = self.processor.get_pred_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        label_list = self.processor.get_labels()
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer, predict_file)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        features = self._get_features_from_input_fn(predict_input_fn, FLAGS.predict_batch_size)

        bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
        assignment_map, probabilities, names = self.model_fn(
                                                      features=features,
                                                      bert_config=bert_config,
                                                      init_checkpoint=FLAGS.init_checkpoint,
                                                      num_labels=len(label_list)
                                                      )
        # for name in names:
        #     print(name)

        # if not self.initialized:
        #     self.sess.run(tf.global_variables_initializer())
        #     self.sess.run(assignment_map)
        #     self.initialized = True
        predictions = []
        while True:
            try:
                probs = self.sess.run([probabilities])[0]
                predictions += list(probs)
                # print(predictions)
            except:
                break
        return np.array(predictions)

    def _validate_features_in_predict_input(self, result):
        if not _has_dataset_or_queue_runner(result):
            logging.warning('Input graph does not use tf.data.Dataset or contain a '
                            'QueueRunner. That means predict yields forever. '
                            'This is probably a mistake.')

    def _get_features_from_input_fn(self, input_fn, batch_size):
        """Extracts the `features` from return values of `input_fn`."""
        params = {"batch_size": batch_size}
        result = input_fn(params)
        iterator = result.make_initializable_iterator()
        self.sess.run(iterator.initializer)
        result = iterator.get_next()
        # result, _, _ = estimator_util.parse_input_fn_result(result)
        self._validate_features_in_predict_input(result)
        return result

    # def create_estimator(self):
    #     bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    #     label_list = self.processor.get_labels()
    #     is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    #     tpu_cluster_resolver = None
    #     run_config = tf.contrib.tpu.RunConfig(
    #         cluster=tpu_cluster_resolver,
    #         master=FLAGS.master,
    #         model_dir=FLAGS.output_dir,
    #         save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #         tpu_config=tf.contrib.tpu.TPUConfig(
    #             iterations_per_loop=FLAGS.iterations_per_loop,
    #             num_shards=FLAGS.num_tpu_cores,
    #             per_host_input_for_training=is_per_host))
    #
    #     num_train_steps = None
    #     num_warmup_steps = None
    #
    #     model_fn = model_fn_builder(
    #         bert_config=bert_config,
    #         num_labels=len(label_list),
    #         init_checkpoint=FLAGS.init_checkpoint,
    #         learning_rate=FLAGS.learning_rate,
    #         num_train_steps=num_train_steps,
    #         num_warmup_steps=num_warmup_steps,
    #         use_tpu=FLAGS.use_tpu,
    #         use_one_hot_embeddings=FLAGS.use_tpu)
    #
    #     # If TPU is not available, this will fall back to normal Estimator on CPU
    #     # or GPU.
    #     estimator = tf.contrib.tpu.TPUEstimator(
    #         use_tpu=FLAGS.use_tpu,
    #         model_fn=model_fn,
    #         config=run_config,
    #         train_batch_size=FLAGS.train_batch_size,
    #         eval_batch_size=FLAGS.eval_batch_size,
    #         predict_batch_size=FLAGS.predict_batch_size)
    #     return estimator

    # def predict(self):
    #     predict_examples = self.processor.get_pred_examples(FLAGS.data_dir)
    #     predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    #     label_list = self.processor.get_labels()
    #     tokenizer = tokenization.FullTokenizer(
    #         vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    #     file_based_convert_examples_to_features(predict_examples, label_list,
    #                                             FLAGS.max_seq_length, tokenizer, predict_file)
    #
    #     predict_drop_remainder = True if FLAGS.use_tpu else False
    #     predict_input_fn = file_based_input_fn_builder(
    #         input_file=predict_file,
    #         seq_length=FLAGS.max_seq_length,
    #         is_training=False,
    #         drop_remainder=predict_drop_remainder)
    #
    #     result = self.estimator.predict(input_fn=predict_input_fn)
    #     return result
