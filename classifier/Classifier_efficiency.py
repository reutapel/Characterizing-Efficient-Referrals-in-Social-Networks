import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import time
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import scipy
import itertools

import logging

# Display progress logs on stdout
LOG_FILENAME = datetime.now().strftime('LogFile_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.INFO,)

# parse commandline arguments
op = OptionParser()
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm", default=True,
              help="Print the confusion matrix.")
op.add_option("--k_fold",
              action='store', type=int, default=100,
              help='k_fold when using cross validation')

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

op.print_help()


###############################################################################
class Classifier:
    def __init__(self):
        self.X_train = None
        self.feature_names = None
        print('{}: Loading the data '.format((time.asctime(time.localtime(time.time())))))
        self.featuresDF = pd.read_excel('FinalFeatures.xlsx')
        self.labels = self.featuresDF['IsEfficient']
        self.submission_author_features = ['submission_author_number_original_subreddit',
                                           'submission_author_number_recommend_subreddit',
                                           'submission_created_time_hour']
        self.sub_comment_author_relation_features = ['cosine_similarity_subreddits_list',
                                                    'comment_submission_similarity',
                                                    'comment_title_similarity']
        self.comment_author_features =['comment_author_number_original_subreddit',
                                        'comment_author_number_recommend_subreddit',
                                        'percent_efficient_references_comment_author',
                                        'number_of_references_comment_author']
        self.comment_features = ['comment_created_time_hour', 'submission_created_time_hour',
                                 'time_between_messages', 'comment_len', 'number_of_r',
                                 'number_of_references_to_submission']
        self.subreddit_features = ['number_of_references_to_recommended_subreddit',
                                   'subreddits_similarity']
        # self.subreddit_features = self.featuresDF['number_of_references_to_recommended_subreddit']
        self.group_dic = {0: [self.submission_author_features, 'submission_author_features'],
                          1: [self.sub_comment_author_relation_features, 'sub_comment_author_relation_features'],
                          2: [self.comment_author_features, 'comment_author_features'],
                          3: [self.comment_features, 'comment_features'],
                          4: [self.subreddit_features, 'subreddit_features']}

        print('{}: Data loaded '.format((time.asctime(time.localtime(time.time())))))
        return

###############################################################################
    def iterateOverFeaturesGroups(self):
        all_groups_results = pd.DataFrame()
        for number_of_groups in range(1, 6):
            for groups in itertools.permutations(range(5), number_of_groups):
                features_group = [self.group_dic[group][0] for group in groups]
                features = [item for sublist in features_group for item in sublist]
                self.X_train = self.featuresDF[features]
                group_names = [self.group_dic[group][1] for group in groups]
                print('{}: Start training with the groups: {} '.format((time.asctime(time.localtime(time.time()))),
                                                                       group_names))
                logging.info('{}: Start training with the groups: {} '
                             .format((time.asctime(time.localtime(time.time()))), group_names))
                group_results = self.ModelsIteration()
                print('{}: Finish training with the groups: {}' \
                    .format((time.asctime(time.localtime(time.time()))), group_names))
                logging.info('{}: Finish training with the groups: {}'
                             .format((time.asctime(time.localtime(time.time()))), group_names))
                # indices = np.arange(len(group_results))
                # results = [[x[i] for x in group_results] for i in range(4)]
                #
                # # clf_names, score, auc, training_time = results
                # clf_names = results[0]
                # score = results[1]
                # auc = results[2]
                # training_time = results[3]
                # training_time = np.array(training_time) / np.max(training_time)
                #
                # plt.figure(figsize=(12, 8))
                # plt.title("Score")
                # plt.barh(indices, score, .2, label="score", color='navy')
                # plt.barh(indices + .3, training_time, .2, label="training time",
                #          color='c')
                # plt.barh(indices, auc, .2, label="ACU", color='darkorange')
                # plt.yticks(())
                # plt.legend(loc='best')
                # plt.subplots_adjust(left=.25)
                # plt.subplots_adjust(top=.95)
                # plt.subplots_adjust(bottom=.05)
                #
                # for i, c in zip(indices, clf_names):
                #     plt.text(-.3, i, c)
                #
                # plt.show()
                # plt.savefig('pythonResults' + group_names + '.png', bbox_inches='tight')

                for model in group_results:
                    model.append(group_names)
                    model.append(opts.k_fold)
                columns_names = ['classifier_name', 'score', 'auc', 'train_time', 'group_list', 'k_fold']
                group_resultsDF = pd.DataFrame(group_results, columns=columns_names)
                # group_results.append(group_names).append([opts.k_fold])
                all_groups_results = all_groups_results.append(group_resultsDF, ignore_index=True)
                all_groups_results.to_csv('pythonResultsTemp.csv', encoding='utf-8')
                # if i == 0:
                #     all_groups_results = group_resultsDF
                #     i += 1
                #     all_groups_results.to_csv('pythonResultsTemp.csv', encoding='utf-8')
                # else:
                #     reut = all_groups_results.append(group_resultsDF, ignore_index=True)
                #     all_groups_results.to_csv('pythonResultsTemp.csv', encoding='utf-8')

        # resultsDF = pd.DataFrame(all_groups_results)
        all_groups_results.to_csv('pythonResultsFinal.csv', encoding='utf-8')

        return


###############################################################################
# benchmark classifiers
    def benchmark(self, clf, clf_name='default'):
        print('_' * 80)
        print('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), clf))
        logging.info('_' * 80)
        logging.info('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), clf))
        t0 = time.time()
        # Cross validation part
        k = opts.k_fold
        if clf_name == 'GaussianNB':
            self.X_train = self.X_train.toarray()
        predicted = cross_val_predict(clf, self.X_train, self.labels, cv=k)
        score = metrics.accuracy_score(self.labels, predicted)
        train_time = time.time() - t0
        print("cross validation time: {}".format(train_time))
        logging.info("cross validation time: {}".format(train_time))
        # if hasattr(clf, 'coef_'):
        #     print("dimensionality: %d" % clf.coef_.shape[1])
        #     print("density: %f" % density(clf.coef_))

            # if opts.print_top10 and self.feature_names is not None:
            #     print("top 10 keywords per class:")
            #     for i, label in enumerate(self.labels):
            #         top10 = np.argsort(clf.coef_[i])[-10:]
            #         print(trim("%s: %s" % (label, " ".join(self.feature_names[top10]))))
            # print()

        # if True:  # opts.print_report:
        #     print("classification report:")
        #     print(metrics.classification_report(self.labels, predicted,
        #                                             self.labels=self.labels))

        if opts.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(self.labels, predicted, labels=[-1, 1]))
            logging.info("confusion matrix:")
            logging.info(metrics.confusion_matrix(self.labels, predicted, labels=[-1, 1]))

            clf_descr = str(clf).split('(')[0]
        print("Accuracy: {} (+/- {})".format(score.mean(), score.std() * 2))
        logging.info("Accuracy: {} (+/- {})".format(score.mean(), score.std() * 2))

        auc = metrics.roc_auc_score(self.labels, predicted, average='samples')
        print('AUC: {}'.format(auc))
        logging.info('AUC: {}'.format(auc))

        return [clf_descr, score, auc, train_time]

    def ModelsIteration(self):
        results = []
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(n_iter=50), "Perceptron"),
                (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                # (RandomForestClassifier(n_estimators=100), "Random forest"),
                (SVC(C=1e-8, gamma=1.0/self.X_train.shape[1], kernel='rbf'), "SVM with RBF Kernel")):

            print('=' * 80)
            print(name)
            results.append(self.benchmark(clf))

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
        results.append(self.benchmark(MultinomialNB(alpha=.01), 'MultinomialNB'))
        results.append(self.benchmark(BernoulliNB(alpha=.01), 'BernoulliNB'))
        # results.append(self.benchmark(GaussianNB(), 'GaussianNB'))

        print('=' * 80)
        print("LinearSVC")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark(LinearSVC(), 'LinearSVC'))

        return results


if __name__ == '__main__':
    classifier = Classifier()
    classifier.iterateOverFeaturesGroups()
