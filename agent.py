from bert.classifier import *

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


def sort_and_retrive(predictions, qa_pairs):
    res = []
    for prediction, qa in zip(predictions, qa_pairs):
        res.append((prediction[1], qa))
    res.sort(reverse=True)
    return res


class PairMatchAgent:
    def __init__(self):
        # qa_file:
        # self.qa_pairs = self.load(qa_file)
        self.classifier = Classifier()
        tf.gfile.MakeDirs(FLAGS.output_dir)
    # def load(self, qa_file):
    #     qa_pairs = []
    #     # this_type = 'q'
    #     q = ''
    #     a = []
    #     id = 1
    #     with open(qa_file, 'r') as fin:
    #         for line in fin.readlines():
    #             if not line or len(line.split()) == 0:
    #                 continue
    #             this_type = line[0]
    #             line = line[1:].strip()
    #             if this_type == 'q':
    #                 if len(a) > 0:
    #                     qa_pairs.append((q, a))
    #                     a = []
    #                 q = line
    #             elif this_type == 'a':
    #                 a.append(line)
    #             else:
    #                 print(id)
    #                 print(this_type)
    #                 print(line)
    #                 raise ValueError('wrong type for line : %s ' % line)
    #             id += 1
    #     qa_pairs.append((q, a))
    #     return qa_pairs

    def record(self, sentence, mode="covered"):
        assert mode in ['covered', 'missed']

        if mode == 'covered':
            with open('covered_records.txt', 'a') as fout:
                fout.write(sentence+'\n')
        else:
            with open('missed_records.txt', 'a') as fout:
                fout.write(sentence+'\n')

    def reply(self, sentence):
        # print('location: agent.reply')
        # return a list of sentences
        qa_pairs = preprocess(sentence)
        # predict based on file generrated by
        predictions = self.classifier.predict()
        ranked_results = sort_and_retrive(predictions, qa_pairs)
        # for k, v in self.qa_pairs:
        #     if k in sentence:
        #         self.record(sentence, mode='covered')
        #         return v

        # self.record(sentence, mode='missed')

        # return ['本宝宝还小，知道的东西不多，这个问题去问人事组的小姐姐们吧～']
        # print('finish agent.reply')
        return ranked_results[0]


# if __name__ == '__main__':
#     agent = PairMatchAgent()
#     while True:
#         msg = input()
#         if msg is None:
#             break
#         ans = agent.reply(msg)
#         for an in ans:
#             print(an)

