import numpy as np
import time
import pandas as pd
import itertools
import random
import gensim
# from time import time
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Doc2Vec
from tqdm import tqdm
from datetime import datetime
tqdm.pandas(desc="progress-bar")
LabeledSentence = gensim.models.doc2vec.LabeledSentence


class TextClassifier:
    def __init__(self):
        print('{}: Loading the data'.format((time.asctime(time.localtime(time.time())))))
        sentences = []
        labels = []
        sentences_len = []
        true_index = []
        false_index = []
        stripComment = lambda x: x.strip().lower()
        replaceComments = lambda x: x.replace(";", ' ').replace(":", ' ').replace('"', ' ').replace('-', ' '). \
            replace(',', ' ').replace('.', ' ').replace("/", ' ').replace('(', ' ').replace(')', ' ')
        splitCommant = lambda x: x.split(" ")
        stop = stopwords.words('english')
        stopWordsComment = lambda x: [i for i in x if i not in stop]
        data = pd.read_excel('FinalFeatures.xlsx')
        comment_index = 0
        for index, comment in data.iterrows():
            train_data = comment['comment_body']
            sentence = stripComment(train_data)
            sentence = replaceComments(sentence)
            sentence = splitCommant(sentence)
            sentence = stopWordsComment(sentence)
            remove_list = []
            for i, word in enumerate(sentence):
                if '\r\r' in word or word == '':
                    remove_list.append(i)
            sentence = [i for j, i in enumerate(sentence) if j not in remove_list]
            sentences.append(sentence)
            labels.append(comment['IsEfficient'])
            sentences_len.append(len(train_data))
            if comment['IsEfficient'] == 1:
                true_index.append(comment_index)
            else:
                false_index.append(comment_index)
            comment_index += 1
        # words = set(itertools.chain(*sentences))

        # choose random index for test set
        true_test_index = random.sample(true_index, 110)
        false_test_index = random.sample(false_index, 740)

        # create test and train sets
        true_test = list(sentences[i] for i in true_test_index)
        true_label = list(labels[i] for i in true_test_index)
        false_test = list(sentences[i] for i in false_test_index)
        false_label = list(labels[i] for i in false_test_index)

        true_train_index = [index for index in true_index if index not in true_test_index]
        false_train_index = [index for index in false_index if index not in false_test_index]
        true_train = list(sentences[i] for i in true_train_index)
        true_train_label = list(labels[i] for i in true_train_index)
        false_train = list(sentences[i] for i in false_train_index)
        false_train_label = list(labels[i] for i in false_train_index)

        X_POS = list(itertools.chain(true_train, true_test))
        # Y_train = list(itertools.chain(true_train_label, false_train_label))
        X_NEG = list(itertools.chain(false_train, false_test))

        X_POS = self.labelizeComments(X_POS, 'POS')
        X_NEG = self.labelizeComments(X_NEG, 'NEG')

        final_sentences = list(itertools.chain(X_POS, X_NEG))

        print('{}: Start calculating Doc2Vec'.format((time.asctime(time.localtime(time.time())))))
        number_of_features = 100
        model = Doc2Vec(min_count=2, window=10, size=number_of_features, negative=5, workers=7, iter=55)
        model.build_vocab(final_sentences)
        #
        print('{}: Start train Doc2Vec'.format((time.asctime(time.localtime(time.time())))))
        for epoch in range(50):
            # model.train(shuffle(final_sentences))
            model.train(final_sentences, total_examples=model.corpus_count, word_count=2)
        #
        model.save('d2v100.d2v')
        # model = Doc2Vec.load('comment.d2v')

        print('{}: Finish calculating Doc2Vec'.format((time.asctime(time.localtime(time.time())))))
        # Create train numpy
        data_size = len(sentences)
        true_size = len(true_train_index) + len(true_test_index)
        false_size = len(false_train_index) + len(false_test_index)
        self.data = np.zeros((data_size, number_of_features))
        self.labels = np.zeros(data_size)

        for i in range(true_size):
            prefix_train_pos = 'POS_' + str(i)
            self.data[i] = model.docvecs[prefix_train_pos]
            self.labels[i] = 1

        j = 0
        for i in range(true_size, true_size + false_size):
            prefix_train_neg = 'NEG_' + str(j)
            self.data[i] = model.docvecs[prefix_train_neg]
            self.labels[i] = -1
            j += 1

        print(self.labels)

        # for Non-Negative values - if we want to train Multinumial NB
        min_max_scale = MinMaxScaler()
        self.data = min_max_scale.fit_transform(self.data)

        comments_id = data['comment_id'].values

        i = 0
        d2v_id = []
        for sample in self.data:
            d2v_id_sample = sample.tolist()
            d2v_id_sample.append(comments_id[i])
            d2v_id.append(d2v_id_sample)
            i += 1

        index = range(number_of_features)
        index.append('comment_id')
        train_vecs_d2vPD = pd.DataFrame.from_records(d2v_id, columns=index)
        final_features = pd.merge(data, train_vecs_d2vPD, on='comment_id')
        final_features.to_csv('100_d2v_scale.csv', encoding='utf-8')

        return

    def labelizeComments(self, comment, label_type):
        labelized = []
        for i, v in tqdm(enumerate(comment)):
            label = '%s_%s' % (label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized


if __name__ == '__main__':
    text_classiier = TextClassifier()