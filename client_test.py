# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model

The SavedModel must be one that can take JPEG images as inputs.


"""

from __future__ import print_function

import base64
import requests
from bert.classifier import *
from agent import preprocess
import json
import numpy as np
import time


# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/bert:predict'


# The image URL is the location of the image we should send to the server
# IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'
def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    serialized = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        # features["input_ids"] = feature.input_ids
        # features["input_mask"] = feature.input_mask
        # features["segment_ids"] = feature.segment_ids
        # features["label_ids"] = [feature.label_id]

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        # writer.write(tf_example.SerializeToString())
        serialized_example = tf_example.SerializeToString()
        serialized.append(serialized_example)
        # features_list.append(features)
    return serialized

# def file_based_convert_examples_to_features(
#         examples, label_list, max_seq_length, tokenizer, output_file):
#     """Convert a set of `InputExample`s to a TFRecord file."""
#     features = collections.OrderedDict()
#     features['input_ids'] = []
#     features['input_mask'] = []
#     features['segment_ids'] = []
#     features['label_ids'] = []
#
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#         feature = convert_single_example(ex_index, example, label_list,
#                                          max_seq_length, tokenizer)
#         features['input_ids'] += feature.input_ids
#         features['input_mask'] += feature.input_mask
#         features['segment_ids'] += feature.segment_ids
#         features['label_ids'] += [feature.label_id]
#
#     def create_int_feature(values):
#         f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
#         return f
#
#     features['input_ids'] = create_int_feature(features['input_ids'])
#     features['input_mask'] = create_int_feature(features['input_mask'])
#     features['segment_ids'] = create_int_feature(features['segment_ids'])
#     features['label_ids'] = create_int_feature(features['label_ids'])
#     tf_example = tf.train.Example(features=tf.train.Features(feature=features))
#     serialized = tf_example.SerializeToString()
#     return serialized


class Client:
    def __init__(self):
        self.processor = MyProcessor()

    def sort_and_retrive(self, predictions, qa_pairs):
        res = []
        for prediction, qa in zip(predictions, qa_pairs):
            res.append((prediction[1], qa))
        res.sort(reverse=True)
        return res

    def preprocess(self, sentence, qa_file='qa_pairs.txt'):
        save_path = os.path.join(FLAGS.data_dir, "pred.tsv")
        qa_path = os.path.join(FLAGS.data_dir, qa_file)
        qa_pairs = []
        with open(save_path, 'w') as fout, open(qa_path, 'r') as fin:
            a = []
            for line in fin.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                if line.startswith('Q:'):
                    # write (0, sentence, candidates) to pred.tsv
                    out_line = '0\t' + sentence + '\t' + line + '\n'
                    fout.write(out_line)

                    if len(a) > 0:
                        qa_pairs.append((q, a))
                    q = line[2:]
                elif line.startswith('A:'):
                    a = [line[2:]]
                else:
                    a.append(line)
            qa_pairs.append((q, a))
        return qa_pairs

    def predict(self, sentence):
        qa_pairs = preprocess(sentence)
        predict_examples = self.processor.get_pred_examples(FLAGS.data_dir)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        label_list = self.processor.get_labels()
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        serialized = file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer, predict_file)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        # predict_input_fn = file_based_input_fn_builder(
        #     input_file=predict_file,
        #     seq_length=FLAGS.max_seq_length,
        #     is_training=False,
        #     drop_remainder=predict_drop_remainder)
        # serialized_examples = tf.data.TFRecordDataset(predict_file)
        # serialized_examples.batch(FLAGS.predict_batch_size)
        # iterator = serialized_examples.make_one_shot_iterator()
        # serialized_example = iterator.get_next()
        # params = {'batch_size': FLAGS.predict_batch_size}
        # features = predict_input_fn(params)
        # iterator = features.make_one_shot_iterator()
        # feature = iterator.get_next()
        # print(type(serialized[0]))

        init = time.time()
        start = init
        # i = 1

        predict_request = '{"instances": ['
        for i in range(len(serialized)):
            if i == 0:
                cur_string = '{"b64": "%s"}' % base64.b64encode(serialized[i]).decode()
            else:
                cur_string = ',{"b64": "%s"}' % base64.b64encode(serialized[i]).decode()
            predict_request += cur_string
        predict_request += ']}'
        response = requests.post(SERVER_URL, data=predict_request)
        # print(response.text)
        response.raise_for_status()
        response = response.json()
        print(response)
        cost = time.time() - init
        print('total time cost: %s s' % cost)

        # for example in serialized:
        #     predict_request = '{"instances" : [{"b64": "%s"}]}' % base64.b64encode(example).decode()
        #     # predict_request = json.dumps({'examples': serialized[0]})
        #     response = requests.post(SERVER_URL, data=predict_request)
        #     # print(response.text)
        #     response.raise_for_status()
        #     prediction = response.json()
        #     prediction = prediction['predictions'][0]
        #     predictions.append(prediction)
        #     cost = time.time() - start
        #     start = time.time()
        #     print('step %s time cost: %s s' % (i, cost))
        #     i += 1
        # cost = time.time()-init
        # print('total time cost: %s s' % cost)
        predictions = response['predictions']
        result = self.sort_and_retrive(predictions=predictions, qa_pairs=qa_pairs)
        return result[0]


# def main():
#   # Download the image
#   dl_request = requests.get(IMAGE_URL, stream=True)
#   dl_request.raise_for_status()
#
#   # Compose a JSON Predict request (send JPEG image in base64).
#   predict_request = '{"instances" : [{"b64": "%s"}]}' % base64.b64encode(
#       dl_request.content)
#
#   # Send few requests to warm-up the model.
#   for _ in range(3):
#     response = requests.post(SERVER_URL, data=predict_request)
#     response.raise_for_status()
#
#   # Send few actual requests and report average latency.
#   total_time = 0
#   num_requests = 10
#   for _ in range(num_requests):
#     response = requests.post(SERVER_URL, data=predict_request)
#     response.raise_for_status()
#     total_time += response.elapsed.total_seconds()
#     prediction = response.json()['predictions'][0]
#
#   print('Prediction class: {}, avg latency: {} ms'.format(
#       prediction['classes'], (total_time*1000)/num_requests))


if __name__ == '__main__':
    client = Client()
    while True:
        msg = input()
        if msg is None:
            break
        prediction = client.predict(msg)
        # print(response)
        print(prediction)
