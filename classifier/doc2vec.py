import numpy as np
from time import time
from datetime import datetime
import time
from nltk.corpus import stopwords
from random import shuffle
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron, LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from gensim.models import Doc2Vec
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale

import pandas as pd
import itertools
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import random
from gensim.models import word2vec
from os.path import join, exists, split
import os
from sklearn.metrics import accuracy_score, auc, roc_curve, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import re
import nltk
import csv
import string
from gensim.models import Phrases
from gensim.models import Word2Vec


def plot_confusion_matrix(data, title='_Confusion Matrix_', cmap=plt.cm.Blues, name=''):

    plt.imshow(data, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    labels = np.array(['Negative', 'Positive'])
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./Plots/" + name + title + '.png', bbox_inches='tight')


def plot_roc_curve(fpr, tpr, roc_auc, name=''):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("./Plots/" + name + '_ROC_Curve.png',
                bbox_inches='tight')


class TextClassifier:
    def __init__(self):
        print('{}: Loading the data'.format((time.asctime(time.localtime(time.time())))))
        # print '{}: Loading the data'.format((time.asctime(time.localtime(time.time()))))
        sentences = []
        labels = []
        sentences_len = []
        true_index = []
        false_index = []
        stripComment = lambda x: x.strip().lower()
        replaceComments = lambda x: x.replace(";", ' ').replace(":", ' ').replace('"', ' ').replace('-', ' ').\
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
        model = Doc2Vec(min_count=2, window=10, size=number_of_features, negative=5, workers=7, iter=55)  # documents=final_sentences,
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
        w2v_id = []
        for sample in self.data:
            w2v_id_sample = sample.tolist()
            w2v_id_sample.append(comments_id[i])
            w2v_id.append(w2v_id_sample)
            i += 1

        index = range(number_of_features)
        index.append('comment_id')
        train_vecs_d2vPD = pd.DataFrame.from_records(w2v_id, columns=index)
        final_features = pd.merge(data, train_vecs_d2vPD, on='comment_id')
        final_features.to_csv('100_d2v_scale.csv', encoding='utf-8')

        return

    def labelizeComments(self, comment, label_type):
        labelized = []
        for i, v in tqdm(enumerate(comment)):
            label = '%s_%s' % (label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

        ###############################################################################
        # benchmark classifiers
    def benchmark(self, model, model_name='default'):
        print('_' * 80)
        print('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), model_name))
        print(model)
        t0 = time.time()
        # Cross validation part
        k = 100
        predicted = cross_val_predict(model, self.data, self.labels, cv=k)
        score = metrics.accuracy_score(self.labels, predicted)
        train_time = time.time() - t0
        print("train and test time: {}".format(train_time))

        print("confusion matrix:")
        print(metrics.confusion_matrix(self.labels, predicted, labels=[-1, 1]))


        model_descr = str(model).split('(')[0]
        print("Accuracy: {} (+/- {})".format(score.mean(), score.std() * 2))

        auc = metrics.roc_auc_score(self.labels, predicted, average='samples')
        print('AUC: {}'.format(auc))

        return [model_descr, score, auc, train_time]

    def ModelsIteration(self):
        results = []
        for model, name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(n_iter=50), "Perceptron"),
                (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(n_estimators=10), "Random forest"),
                (SVC(C=1e-8, kernel='rbf'), "SVM with RBF Kernel")):

            print('=' * 80)
            print(name)
            results.append(self.benchmark(model, name))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            results.append(self.benchmark(LinearSVC(loss='squared_hinge', penalty=penalty,
                                                    dual=False, tol=1e-3), 'LinearSVC'))

            # Train SGD model
            results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty), 'SGDClassifier'))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        # results.append(self.benchmark(MultinomialNB(alpha=.01), 'MultinomialNB'))
        results.append(self.benchmark(BernoulliNB(alpha=.01), 'BernoulliNB'))
        # results.append(self.benchmark(GaussianNB(), 'GaussianNB'))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark(LinearSVC(), 'classification'))

        print('=' * 80)
        print('Logistic Regression')
        results.append(self.benchmark(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                                         intercept_scaling=1, penalty='l2', random_state=None,
                                                         tol=0.0001), 'Logistic Regression'))

        return results


if __name__ == '__main__':
    text_classiier = TextClassifier()
    text_classiier.ModelsIteration()
