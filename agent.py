from bert.classifier import *
import base64
import requests


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer):
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


def preprocess(sentence, qa_file='qa_pairs.txt'):
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


class PairMatchAgent:
    def __init__(self, server_url='http://localhost:8501/v1/models/bert:predict'):
        self.processor = MyProcessor()
        self.server_url = server_url

    def sort_and_retrive(self, predictions, qa_pairs):
        res = []
        for prediction, qa in zip(predictions, qa_pairs):
            res.append((prediction[1], qa))
        res.sort(reverse=True)
        return res

    def create_request(self, serialized):
        predict_request = '{"instances": ['
        for i in range(len(serialized)):
            if i == 0:
                cur_string = '{"b64": "%s"}' % base64.b64encode(serialized[i]).decode()
            else:
                cur_string = ',{"b64": "%s"}' % base64.b64encode(serialized[i]).decode()
            predict_request += cur_string
        predict_request += ']}'
        return predict_request

    def predict(self, sentence):
        qa_pairs = preprocess(sentence)
        predict_examples = self.processor.get_pred_examples(FLAGS.data_dir)
        label_list = self.processor.get_labels()
        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        serialized = file_based_convert_examples_to_features(predict_examples, label_list,
                                                             FLAGS.max_seq_length, tokenizer)

        predict_request = self.create_request(serialized)
        response = requests.post(self.server_url, data=predict_request)
        response.raise_for_status()
        response = response.json()
        predictions = response['predictions']
        result = self.sort_and_retrive(predictions=predictions, qa_pairs=qa_pairs)
        return result[0]

# if __name__ == '__main__':
#     agent = PairMatchAgent()
#     while True:
#         msg = input()
#         if msg is None:
#             break
#         ans = agent.reply(msg)
#         for an in ans:
#             print(an)
