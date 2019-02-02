from optparse import OptionParser
import sys
from time import time
import time
from datetime import datetime
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.svm import SVC
import itertools
import logging
import os
import math

# Display progress logs on stdout
base_directory = os.path.abspath(os.curdir)
logs_directory = os.path.join(base_directory, 'new_logs')
LOG_FILENAME = os.path.join(logs_directory, datetime.now().strftime
('LogFile_100w2v_scale_newCV_no_Peff_Split_TimeComment_perFeature_stepwise_backward_causality_data_%d%m%Y%H%M.log'))
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO,)

# parse commandline arguments
op = OptionParser()
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm", default=True,
              help="Print the confusion matrix.")
op.add_option("--k_fold",
              action='store', type=int, default=15,
              help='k_fold when using cross validation')
op.add_option("--split_Peff",
              action='store', default=True,
              help='whether to create classifier for each Peff group')
op.add_option("--is_backward",
              action='store', default=True,
              help='whether to use backward elimination or forward selection, if True - use backward elimination')

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()


###############################################################################
class Classifier:
    def __init__(self):
        self.X_train = None
        self.features = None
        self.feature_names = None
        print('{}: Loading the data: 100w2v_scale_2_causality'.
              format((time.asctime(time.localtime(time.time())))))
        self.original_data = pd.read_excel('100w2v_scale_2_causality.xlsx')
        self.labels = None
        self.featuresDF = None

        # for 50Doc2Vec:
        # self.text_features = range(50)
        # for Word2Vec and 100Doc2Vec:
        self.text_features = range(100)

        self.group_dic = {0: [['submission_author_number_original_subreddit'],
                              'submission_author_number_original_subreddit'],
                          1: [['submission_author_number_recommend_subreddit'],
                              'submission_author_number_recommend_subreddit'],
                          2: [['submission_created_time_hour'], 'submission_created_time_hour'],
                          3: [['cosine_similarity_subreddits_list'], 'cosine_similarity_subreddits_list'],
                          4: [['comment_submission_similarity'], 'comment_submission_similarity'],
                          5: [['comment_title_similarity'], 'comment_title_similarity'],
                          6: [['comment_author_number_original_subreddit'], 'comment_author_number_original_subreddit'],
                          7: [['comment_author_number_recommend_subreddit'], 'comment_author_number_recommend_subreddit'],
                          8: [['number_of_references_comment_author'], 'number_of_references_comment_author'],
                          9: [['comment_created_time_hour'], 'comment_created_time_hour'],
                          10: [['time_between_messages'], 'time_between_messages'],
                          11: [['comment_len'], 'comment_len'],
                          12: [['number_of_r'], 'number_of_r'],
                          13: [['number_of_references_to_submission'], 'number_of_references_to_submission'],
                          14: [['number_of_references_to_recommended_subreddit'],
                               'number_of_references_to_recommended_subreddit'],
                          15: [['subreddits_similarity'], 'subreddits_similarity'],
                          16: [['treated'], 'treated']
                          # 16: [self.text_features, 'text_features']
                          }

        print('{}: Data loaded '.format((time.asctime(time.localtime(time.time())))))
        return

###############################################################################
    def split_relevant_data(self, Peff_up_threshold, Peff_down_threshold):
        self.featuresDF = self.original_data.loc[(self.original_data['percent_efficient_references_comment_author'] <=
                                                  Peff_up_threshold) &
                                                 (self.original_data['percent_efficient_references_comment_author'] >=
                                                  Peff_down_threshold)]
        # Split the data to k=15 groups, each comment_author in one group only
        i = 0
        number_sample_group = 0
        if Peff_up_threshold == 50.0 or Peff_up_threshold == 60.0 or Peff_up_threshold == 100.0:
            opts.k_fold = 4
        sample_per_group = self.featuresDF.shape[0] / opts.k_fold
        last_comment_author = ''
        for index, row in self.featuresDF.iterrows():
            if number_sample_group < sample_per_group:
                self.featuresDF.set_value(index, 'group_number', i)
                number_sample_group += 1
                last_comment_author = row['comment_author']
            else:
                if last_comment_author != row['comment_author']:
                    i += 1
                    self.featuresDF.set_value(index, 'group_number', i)
                    print('{}: finish split samples for group number {} with {} samples'.
                          format((time.asctime(time.localtime(time.time()))), i-1, number_sample_group))
                    print('{}: start split samples for group number {}'.
                          format((time.asctime(time.localtime(time.time()))), i))
                    logging.info('{}: finish split samples for group number {} with {} samples'.
                                 format((time.asctime(time.localtime(time.time()))), i-1, number_sample_group))
                    logging.info('{}: start split samples for group number {}'.
                                 format((time.asctime(time.localtime(time.time()))), i))
                    last_comment_author = row['comment_author']
                    number_sample_group = 1
                else:
                    self.featuresDF.set_value(index, 'group_number', i)
                    number_sample_group += 1
                    last_comment_author = row['comment_author']
                    print('{}: {} group is larger, number of samples is: {}'.
                          format((time.asctime(time.localtime(time.time()))), i, number_sample_group))
        print('{}: finish split samples for group number {} with {} samples'.
              format((time.asctime(time.localtime(time.time()))), i, number_sample_group))
        logging.info('{}: finish split samples for group number {} with {} samples'.
                     format((time.asctime(time.localtime(time.time()))), i, number_sample_group))
        opts.k_fold = i + 1
        self.labels = self.featuresDF[['IsEfficient', 'group_number']]
        print('{}: Finish split the data for Peff between: {} and {}'.
              format((time.asctime(time.localtime(time.time()))), Peff_down_threshold, Peff_up_threshold))
        logging.info('{}: Finish split the data for Peff between: {} and {}'.
                     format((time.asctime(time.localtime(time.time()))), Peff_down_threshold, Peff_up_threshold))


###############################################################################
    def iterateOverFeaturesGroups(self, Peff_up_threshold, Peff_down_threshold):
        all_groups_results = pd.DataFrame()
        remaining_features = list(self.group_dic.keys())
        if opts.is_backward:  # use backward elimination
            selected_features = list(self.group_dic.keys())
        else:  # use forward selection
            selected_features = []
            remaining_features = [x for x in remaining_features if x not in selected_features]
        current_auc, best_new_auc = 0.0, 0.0
        remain_number_of_candidate = len(remaining_features)
        while remaining_features and current_auc == best_new_auc and remain_number_of_candidate > 0:
            auc_with_candidates = list()
            for candidate in remaining_features:
                if opts.is_backward:  # use backward elimination
                    features_group = [self.group_dic[group][0] for group in selected_features]
                    features_group.remove(self.group_dic[candidate][0])
                    self.features = [item for sublist in features_group for item in sublist]
                    features = [item for sublist in features_group for item in sublist]
                    features.append('group_number')
                    self.X_train = self.featuresDF[features]
                    features_names = [self.group_dic[feature][1] for feature in selected_features]
                    features_names.remove(self.group_dic[candidate][1])

                else:  # use forward selection
                    features_group = [self.group_dic[group][0] for group in selected_features] +\
                                     [self.group_dic[candidate][0]]
                    self.features = [item for sublist in features_group for item in sublist]
                    features = [item for sublist in features_group for item in sublist]
                    features.append('group_number')
                    self.X_train = self.featuresDF[features]
                    features_names = [self.group_dic[feature][1] for feature in selected_features] +\
                                     [self.group_dic[candidate][1]]

                print('{}: Start training with the groups: {} '.format((time.asctime(time.localtime(time.time()))),
                                                                       features_names))
                logging.info('{}: Start training with the groups: {} '
                             .format((time.asctime(time.localtime(time.time()))), features_names))
                group_results = self.ModelsIteration()
                best_auc = max(result[2] for result in group_results)
                auc_with_candidates.append((best_auc, candidate))

                print('{}: Finish training with the groups: {}'.
                      format((time.asctime(time.localtime(time.time()))), features_names))
                logging.info('{}: Finish training with the groups: {}'.
                             format((time.asctime(time.localtime(time.time()))), features_names))

                for model in group_results:
                    model.append(features_names)
                    model.append(opts.k_fold)
                    model.append(Peff_up_threshold)
                    model.append(Peff_down_threshold)
                columns_names = ['classifier_name', 'score', 'auc', 'train_time', 'features_list', 'k_fold',
                                 'Peff_up_threshold', 'Peff_down_threshold']
                group_resultsDF = pd.DataFrame(group_results, columns=columns_names)
                # group_results.append(group_names).append([opts.k_fold])
                all_groups_results = all_groups_results.append(group_resultsDF, ignore_index=True)
                all_groups_results.to_csv('test_results_stepwise.csv', encoding='utf-8')

            auc_with_candidates.sort()
            best_new_auc, best_candidate = auc_with_candidates.pop()
            if current_auc <= best_new_auc:
                if opts.is_backward:  # use backward elimination
                    selected_features.remove(best_candidate)
                else:  # use forward selection
                    selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
                current_auc = best_new_auc

            else:
                logging.info('{}: No candidate was chosen for threshold: {} and {}, number of selected features is {}.'.
                             format((time.asctime(time.localtime(time.time()))), Peff_down_threshold, Peff_up_threshold,
                                    len(selected_features)))
                print('{}: No candidate was chosen for threshold: {} and {}, number of selected features is {}.'.
                      format((time.asctime(time.localtime(time.time()))), Peff_down_threshold, Peff_up_threshold,
                             len(selected_features)))

            # one candidate can be chosen, if not- we go forward to the next step.
            remain_number_of_candidate -= 1

        selected_features_names = [self.group_dic[feature][1] for feature in selected_features]
        logging.info('{}: Selected features for threshold: {} and {} are: {} and the best AUC is: {}'.
                     format((time.asctime(time.localtime(time.time()))), Peff_down_threshold, Peff_up_threshold,
                            selected_features_names, best_new_auc))
        print('{}: Selected features for threshold: {} and {} are: {} and the best AUC is: {}.'.
              format((time.asctime(time.localtime(time.time()))), Peff_down_threshold, Peff_up_threshold,
                     selected_features_names, best_new_auc))

        return all_groups_results


###############################################################################
# benchmark classifiers
    def benchmark(self, clf, clf_name='default'):
        # if I want to train only specific model:
        # if clf_name != 'MultinomialNB':
        #     print('Not training')
        #     return ['not training', 0, 0, 0]
        print('_' * 80)
        print('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), clf))
        logging.info('_' * 80)
        logging.info('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), clf))
        # Cross validation part
        if clf_name == 'GaussianNB':
            self.X_train = self.X_train.toarray()
        t1 = time.time()
        score = []
        auc = []
        for out_group in range(opts.k_fold):
            t0 = time.time()
            # create train and test data
            test_data = self.X_train.loc[self.X_train['group_number'] == out_group][self.features]
            test_label = self.labels.loc[self.X_train['group_number'] == out_group]['IsEfficient']
            train_data = self.X_train.loc[self.X_train['group_number'] != out_group][self.features]
            train_label = self.labels.loc[self.X_train['group_number'] != out_group]['IsEfficient']

            # train the model
            clf.fit(train_data, train_label)
            predicted = clf.predict(test_data)
            score.append(metrics.accuracy_score(test_label, predicted))
            auc.append(metrics.roc_auc_score(test_label, predicted, average='samples'))
            # print('fold number {}: accuracy: {}, AUC: {}'.format(out_group, metrics.accuracy_score(test_label,
            #                                                                                        predicted),
            #                                                      metrics.roc_auc_score(test_label, predicted,
            #                                                                            average='samples')))

            logging.info("Fold number:")
            logging.info(out_group)
            logging.info("accuracy:")
            logging.info(metrics.accuracy_score(test_label, predicted))
            logging.info("AUC:")
            logging.info(metrics.roc_auc_score(test_label, predicted, average='samples'))
            if opts.print_cm:
                # print("confusion matrix:")
                # print(metrics.confusion_matrix(test_label, predicted, labels=[-1, 1]))
                logging.info("confusion matrix:")
                logging.info(metrics.confusion_matrix(test_label, predicted, labels=[-1, 1]))
            train_time = time.time() - t0
            # print("fold number {}: cross validation time: {}".format(out_group, train_time))
            logging.info("cross validation time: {}".format(train_time))

        # clf_descr = str(clf).split('(')[0]
        average_acc = sum(score)/len(score)
        print("Average Accuracy: {}".format(average_acc))
        logging.info("Average Accuracy: {})".format(average_acc))

        average_auc = sum(auc)/len(auc)
        print("Average AUC: {}".format(average_auc))
        logging.info('Average AUC: {}'.format(average_auc))

        train_time = time.time() - t1

        return [clf_name, average_acc, average_auc, train_time]

    def ModelsIteration(self):
        results = []
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(max_iter=1000), "Perceptron"),
                (PassiveAggressiveClassifier(max_iter=1000), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(), 'Random forest'),  # n_estimators=100, bootstrap=False
                (SVC(C=1e-8, gamma=1.0/self.X_train.shape[1], kernel='rbf'), "SVM with RBF Kernel")):

            print('=' * 80)
            print(name)
            results.append(self.benchmark(clf, name))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            results.append(self.benchmark(LinearSVC(loss='squared_hinge', penalty=penalty,
                                                    dual=False, tol=1e-3), 'LinearSVC_' + penalty))

            # Train SGD model
            results.append(self.benchmark(SGDClassifier(alpha=.0001, max_iter=1000, penalty=penalty),
                                          'SGDClassifier_' + penalty))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(self.benchmark(SGDClassifier(alpha=.0001, max_iter=1000, penalty="elasticnet"),
                       "Elastic-Net penalty"))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark(NearestCentroid(), 'NearestCentroid'))

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

        print('=' * 80)
        print('Logistic Regression')
        results.append(self.benchmark(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                                         intercept_scaling=1, penalty='l2', random_state=None,
                                                         tol=0.0001), 'Logistic Regression'))

        return results


if __name__ == '__main__':
    classifier = Classifier()
    # threshold_list = [0.0, 20.0, 60.0, 100.0]
    threshold_list = [60.0, 100.0]
    all_Peff_groups_results = pd.DataFrame()
    for i in range(len(threshold_list) - 1):
        classifier.split_relevant_data(threshold_list[i+1], threshold_list[i])
        all_groups_results = classifier.iterateOverFeaturesGroups(threshold_list[i+1], threshold_list[i])
        all_Peff_groups_results = all_Peff_groups_results.append(all_groups_results, ignore_index=True)

    classifiers_list = all_Peff_groups_results.classifier_name.unique()
    groups_list = all_Peff_groups_results['features_list'].tolist()
    groups_list.sort()
    groups_list = [k for k, _ in itertools.groupby(groups_list)]
    classifiers_results_dict = dict()
    for i in classifiers_list:
        for j in groups_list:
            key = (i, tuple(j))
            classifiers_results_dict[key] = []
            classifiers_results_dict[key].append([])
            classifiers_results_dict[key].append([])

    for index, row in all_Peff_groups_results.iterrows():
        classifiers_results_dict[(row['classifier_name'], tuple(row['features_list']))][1].append(row['auc'])
        classifiers_results_dict[(row['classifier_name'], tuple(row['features_list']))][0].append(row['score'])

    average_results_all_models = list()
    for key in classifiers_results_dict.keys():
        score_value = classifiers_results_dict[key][0]
        auc_value = classifiers_results_dict[key][1]
        average_result = [key[0], sum(score_value) / float(len(score_value)), sum(auc_value) / float(len(auc_value)),
                          0.0, key[1], 'average', 0.0, 0.0]
        average_results_all_models.append(average_result)
    columns_names = ['classifier_name', 'score', 'auc', 'train_time', 'features_list', 'k_fold',
                     'Peff_up_threshold', 'Peff_down_threshold']
    group_resultsDF = pd.DataFrame(average_results_all_models, columns=columns_names)

    all_Peff_groups_results = all_Peff_groups_results.append(average_results_all_models, ignore_index=True)
    all_Peff_groups_results.to_csv('results_final_split_no_Peff_time_per_feature_stepwise_backward_60100_causality.csv',
                                   encoding='utf-8')
