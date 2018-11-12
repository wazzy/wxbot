import tensorflow as tf
import os
from bert.classifier import *

classifier = Classifier()


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


# def process_input(sentence, target_file='targets.txt'):
#     save_path = os.path.join(FLAGS.data_dir, "pred.tsv")
#     target_path = os.path.join(FLAGS.data_dir, target_file)
#     with open(save_path, 'w') as fout, open(target_path, 'r') as fin:
#         for line in fin.readlines():
#             out_line = '0\t'+sentence+'\t'+line
#             fout.write(out_line)


# def sort_and_retrive(predictions, target_file='targets.txt'):
#     res = []
#     predictions = list(predictions)
#     target_path = os.path.join(FLAGS.data_dir, target_file)
#     with open(target_path, 'r') as fin:
#         lines = list(fin.readlines())
#         assert len(predictions) == len(lines), 'number of predictions and target lines not equal'
#         for prediction, line in zip(predictions, lines):
#             res.append((prediction[1], line.strip()))
#     res.sort(reverse=True)
#     return res

def sort_and_retrive(predictions, qa_pairs):
    res = []
    for prediction, qa in zip(predictions, qa_pairs):
        res.append((prediction[1], qa))
    res.sort(reverse=True)
    return res

tf.gfile.MakeDirs(FLAGS.output_dir)

while True:
    sentence = input()
    if sentence is None or len(sentence.split()) == 0:
        break
    qa_pairs = preprocess(sentence)
    predictions = classifier.predict_online()
    ranked_results = sort_and_retrive(predictions, qa_pairs)
    print(ranked_results[0][1])
