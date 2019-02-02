import pandas as pd
import numpy as np
from time import time
import time
import itertools
import math
from collections import Counter
from time import strftime, localtime
from datetime import datetime
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from optparse import OptionParser
import sys
from sklearn.metrics.pairwise import cosine_similarity
import pickle


op = OptionParser()
op.add_option('--efficient_threshold',
              action='store', type=int, default=0.8,
              help='threshold for efficient reference according to classifier score')
op.add_option('--use_date_threshold',
              action='store', default=True,
              help='take only data about comments and submissions that published before the comment')
op.add_option('--pickel_not_saved',
              action='store', default=False,
              help='did I have already saved the subreddit_dict into pickel --> just need to load it')

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


class CreateFeatures:
    def __init__(self):
        self.classify_ref = pd.read_excel('train_vecs_w2vPD.xlsx')  # 'FinalResultsWithEfficient_13.4.17_8.csv, Features.xlsx'
        pd.to_numeric(self.classify_ref['submission_created_time'])
        pd.to_numeric(self.classify_ref['comment_created_time'])
        # self.classify_ref.sort_values(by=['IsEfficient'], ascending=False, axis=0)
        self.all_data = pd.read_excel('all_data.xlsx')  # 'before_filtering_1447.csv'
        pd.to_numeric(self.all_data['comment_created_time'])
        pd.to_numeric(self.all_data['submission_created_time'])
        # self.all_data.sort_values(by=['comment_author'], ascending=False, axis=0)
        self.references = pd.read_excel('resultsPredicted.xlsx')  # ('resultsPredicted1600.csv')
        self.references.sort_values(by=['classifier_result'], ascending=False, axis=0)
        pd.to_numeric(self.references['submission_created_time'])
        pd.to_numeric(self.references['comment_created_time'])
        self.subreddit_dict = dict()

    # Get the number of messages the user wrote in each of the subreddit and the list of subreddits he participant in
    def number_list_of_message(self, subreddit1, subreddit2, user, comment_time):
        # Create dataframes with author and subreddit- no duplicates, all the messages (submission and comment)
        # the author has published
        submission_data = self.all_data[['submission_author', 'subreddit', 'submission_id', 'submission_created_time']]
        submission_data = submission_data.drop_duplicates()
        submission_data.columns = ['author', 'subreddit', 'id', 'time']
        comment_data = self.all_data[['comment_author', 'subreddit', 'comment_id', 'comment_created_time']]
        comment_data = comment_data.drop_duplicates()
        comment_data.columns = ['author', 'subreddit', 'id', 'time']
        merge_results = pd.concat([submission_data, comment_data])
        number_subreddit1 = len(merge_results[
            (merge_results['author'] == user) & (merge_results['subreddit'] == subreddit1) &
            (merge_results['time'] < comment_time)])
        number_subreddit2 = len(merge_results[
            (merge_results['author'] == user) & (merge_results['subreddit'] == subreddit2) &
            (merge_results['time'] < comment_time)])
        user_subreddits_list = list(merge_results[(merge_results['author'] == user) &
                                                  (merge_results['time'] < comment_time)]['subreddit'])
        return number_subreddit1, number_subreddit2, user_subreddits_list

    # Get the number of references the comment user wrote, reference is according to the first classifier results.
    def number_of_references(self, user, comment_time):
        number_of_references = len(self.references[
                                      (self.references['comment_author'] == user) &
                                      (self.references['classifier_result'] >= opts.efficient_threshold) &
                                      (self.references['comment_created_time'] < comment_time)])
        # number_of_references_1 = 0
        # for index, comment in self.references.iterrows():
        #     if comment['comment_author'] == user:
        #         if comment['classifier_result'] >= 0.8:
        #             number_of_references_1 += 1
        return number_of_references

    # Get the number of references the comment use wrote and we checked their efficiency.
    def number_of_checked_references(self, user, comment_time):
        number_of_checked_references = len(self.classify_ref[
                                               (self.classify_ref['comment_author'] == user) &
                                                (self.classify_ref['comment_created_time'] < comment_time)])
        return number_of_checked_references

    # Get the number of efficiency references the comment user wrote
    def number_of_efficient_references(self, user, comment_time, is_efficient):
        number_of_efficient_references = len(self.classify_ref[
                                         (self.classify_ref['comment_author'] == user) &
                                         (self.classify_ref['IsEfficient'] == is_efficient) &
                                         (self.classify_ref['comment_created_time'] < comment_time)])
        return number_of_efficient_references

    # Get the number of references to the recommended subreddit.
    def popular_subreddit(self, recommended_subreddit, comment_time):
        number_of_references = len(self.references[
                                                   (self.references['recommend_subreddit'] == recommended_subreddit) &
                                                   (self.references['classifier_result'] >= opts.efficient_threshold) &
                                                   (self.references['comment_created_time'] < comment_time)])
        return number_of_references

    # Create the subreddit_dict to use later in the code
    def create_subreddit_data(self):
        print('{}: Start calculate subreddit dictionary'.format((time.asctime(time.localtime(time.time())))))
        for index, comment in self.all_data.iterrows():
            title = comment['title']
            if isinstance(title, str):
                title.encode('utf-8')
                if not isinstance(title, str) or title in ['[removed]', '[deleted]']:
                    title = ' '
            else:
                title = ' '
            submission_body = comment['submission_body']
            if isinstance(submission_body, str):
                submission_body.encode('utf-8')
                if not isinstance(submission_body, str) or submission_body in ['[removed]', '[deleted]']:
                    submission_body = ' '
            else:
                submission_body = ' '
            comment_body = comment['comment_body']
            if isinstance(comment_body, str):
                comment_body.encode('utf-8')
                if not isinstance(comment_body, str) or comment_body in ['[removed]', '[deleted]']:
                    comment_body = ' '
            else:
                comment_body = ' '

            concat_text = title + ' ' + submission_body + ' ' + comment_body
            subreddit = comment['subreddit']
            if isinstance(subreddit, str):
                subreddit.encode('utf-8')
            if subreddit in self.subreddit_dict.keys():
                self.subreddit_dict[subreddit] = self.subreddit_dict[subreddit] + concat_text
            else:
                self.subreddit_dict[subreddit] = concat_text
        with open('subreddit_dict.pickle', 'wb') as handle:
            pickle.dump(self.subreddit_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('{}: Finish calculate and save subreddit dictionary'.format((time.asctime(time.localtime(time.time())))))
        return

    # Calculate the cosine similarity of the tfidf vectors of 2 texts.
    def tfifd_similarity(self, text, text2=None):
        # in case we want the similarity between 2 subreddits text
        if text2 is not None:
            if text in self.subreddit_dict.keys() and text2 in self.subreddit_dict.keys():
                text = [self.subreddit_dict[text], self.subreddit_dict[text2]]
            else:
                return 0
        tfidf = TfidfVectorizer().fit_transform(text)
        similarity = cosine_similarity(tfidf[0:], tfidf[1:])
        return similarity[0][0]


# Calculate the cosine similarity between 2 vectors
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# Convert utc time to regular format of time
def convert_utc(utc_time):
    tz = pytz.timezone('GMT')  # America/New_York
    dt = datetime.fromtimestamp(utc_time, tz)
    return dt


# Count the number of subreddits in the reference and its length.
def number_of_subreddits(comment, string_to_find):
    # comment_body = comment.encode('utf-8')
    comment_body = comment
    reference_subreddit = comment_body[comment_body.find(string_to_find):].split('/')  # split the body from /r/
    number_of_r = 0
    comment_len = len(comment_body)
    comment_list_len = len(reference_subreddit)
    for i in range(0, comment_list_len):
        if reference_subreddit[i] == 'r':
            number_of_r += 1
    return comment_len, number_of_r


def main(only_subreddit_similarity=False, only_percent=False):
    print('{}: Loading the data'.format((time.asctime(time.localtime(time.time())))))
    create_features = CreateFeatures()
    print('{}: Finish loading the data'.format((time.asctime(time.localtime(time.time())))))
    print('data sizes: all data: {}, ref data: {}, classify ref data: {} '.format(create_features.all_data.shape,
                                                                                  create_features.references.shape,
                                                                                  create_features.classify_ref.shape))
    if opts.pickel_not_saved:
        create_features.create_subreddit_data()
    else:
        with open('subreddit_dict.pickle', 'rb') as handle:
            create_features.subreddit_dict = pickle.load(handle)

    all_comments_features = list()
    for index, comment in create_features.classify_ref.iterrows():
        if index % 100 == 0:
            print('{}: Finish calculate {} samples'.format((time.asctime(time.localtime(time.time()))), index))
        comment_author = comment['comment_author']
        original_subreddit = comment['subreddit']
        recommend_subreddit = comment['recommend_subreddit']
        if opts.use_date_threshold:  # if we use the data threshold - use the comment time, else use the current time.
            comment_time = comment['comment_created_time']
            submission_time = comment['submission_created_time']
        else:
            comment_time = datetime.utcnow()
            submission_time = datetime.utcnow()

        if only_subreddit_similarity:
            subreddits_similarity = create_features.tfifd_similarity(original_subreddit, recommend_subreddit)
            featuresDF = pd.Series(subreddits_similarity)
        elif only_percent:
            number_of_efficient_references_comment_author = \
                create_features.number_of_efficient_references(comment_author, comment_time)
            number_of_checked_references = create_features.number_of_checked_references(comment_author, comment_time)
            if number_of_checked_references > 0:
                percent_efficient_references_comment_author = (100.0 * number_of_efficient_references_comment_author) / \
                                                              number_of_checked_references
            else:
                percent_efficient_references_comment_author = 0
                # print('percent_efficient_references_comment_author is 0 for comment ID: {}'.format(comment['comment_id']))
            featuresDF = pd.Series(percent_efficient_references_comment_author)
        else:
            # Calculate similarity between the original and recommended subreddits:
            subreddits_similarity = create_features.tfifd_similarity(original_subreddit, recommend_subreddit)
            # Get comment author features:
            comment_author_number_original_subreddit, comment_author_number_recommend_subreddit, \
            comment_author_subreddit_list = create_features.number_list_of_message(original_subreddit,
                                                                                   recommend_subreddit,
                                                                                   comment_author, comment_time)
            number_of_references_comment_author = create_features.number_of_references(comment_author, comment_time)
            # print('{}: comment ID: {}, number_of_references_comment_author: {}'\
            #     .format((time.asctime(time.localtime(time.time()))), comment['comment_id'],
            #             number_of_references_comment_author))
            number_of_efficient_references_comment_author = \
                create_features.number_of_efficient_references(comment_author, comment_time, is_efficient=1)
            number_of_inefficient_references_comment_author = \
                create_features.number_of_efficient_references(comment_author, comment_time, is_efficient=-1)
            number_of_checked_references = create_features.number_of_checked_references(comment_author, comment_time)
            if number_of_checked_references > 0:
                percent_efficient_references_comment_author = (100.0 * number_of_efficient_references_comment_author) / \
                                                              number_of_checked_references
            else:
                percent_efficient_references_comment_author = 0
                # print('percent_efficient_references_comment_author is 0 for comment ID: {}'.format(comment['comment_id']))
            # Get submission author features:
            submission_author = comment['submission_author']
            submission_author_number_original_subreddit, submission_author_number_recommend_subreddit, \
            submission_author_subreddit_list = create_features.number_list_of_message(original_subreddit,
                                                                                      recommend_subreddit,
                                                                                      submission_author, submission_time)
            # Similarity between comment and submission authors subreddits lists:
            cosine_similarity_subreddits_list = get_cosine(Counter(comment_author_subreddit_list),
                                                           Counter(submission_author_subreddit_list))
            # Get the hour of the comment and the submission:
            comment_created_time_hour = convert_utc(comment['comment_created_time']).hour
            submission_created_time_hour = convert_utc(comment['submission_created_time']).hour
            # Get the time between the submission was published and the comment time:
            time_to_comment = comment['time_to_comment']
            time_between_messages_hour = math.floor(time_to_comment/3600.0)
            time_between_messages_min = math.floor((time_to_comment - 3600*time_between_messages_hour)/60.0)/100.0
            time_between_messages = time_between_messages_hour + time_between_messages_min
            # Comment features:
            comment_body = comment['comment_body']
            submission_body = comment['submission_body']
            submission_title = comment['title']
            comment_len, number_of_r = number_of_subreddits(comment_body, '/r/')
            if isinstance(submission_body, str) and isinstance(comment_body, str):
                comment_submission_similarity = create_features.tfifd_similarity([comment_body, submission_body])
            else:
                comment_submission_similarity = 0.0
            if isinstance(submission_title, str) and isinstance(comment_body, str):
                comment_title_similarity = create_features.tfifd_similarity([comment_body, submission_title])
            else:
                comment_title_similarity = 0.0
            number_of_references_to_submission = comment['num_comments']
            # subreddit features:
            number_of_references_to_recommended_subreddit = create_features.popular_subreddit(recommend_subreddit,
                                                                                              comment_time)

            features = [comment_author_number_original_subreddit,
                        comment_author_number_recommend_subreddit, percent_efficient_references_comment_author,
                        number_of_references_comment_author, number_of_efficient_references_comment_author,
                        number_of_inefficient_references_comment_author,
                        submission_author_number_original_subreddit, submission_author_number_recommend_subreddit,
                        cosine_similarity_subreddits_list, comment_created_time_hour, submission_created_time_hour,
                        time_between_messages, comment_len, number_of_r, comment_submission_similarity,
                        comment_title_similarity, number_of_references_to_submission,
                        number_of_references_to_recommended_subreddit, subreddits_similarity]
            labels = ('comment_author_number_original_subreddit', 'comment_author_number_recommend_subreddit',
                      'percent_efficient_references_comment_author', 'number_of_references_comment_author',
                      'number_of_efficient_references_comment_author', 'number_of_inefficient_references_comment_author',
                      'submission_author_number_original_subreddit',
                      'submission_author_number_recommend_subreddit', 'cosine_similarity_subreddits_list',
                      'comment_created_time_hour', 'submission_created_time_hour', 'time_between_messages',
                      'comment_len', 'number_of_r', 'comment_submission_similarity', 'comment_title_similarity',
                      'number_of_references_to_submission', 'number_of_references_to_recommended_subreddit',
                      'subreddits_similarity')

            featuresDF = pd.Series(features, index=labels)

        comment_features = comment.append(featuresDF)
        if only_subreddit_similarity:
            comment_features.rename(columns={'0': 'subreddits_similarity'}, inplace=True)
        elif only_percent:
            comment_features.rename(columns={'0': 'percent_efficient_references_comment_author'}, inplace=True)

        if index == 0:
            all_comments_features = comment_features
            # print('{}: Finish calculate first samples'.format((time.asctime(time.localtime(time.time())))))
        else:
            all_comments_features = pd.concat([comment_features, all_comments_features], axis=1)

        all_comments_features.T.to_csv('Features_with_commnent_time.csv', encoding='utf-8')

    # export the data to csv file
    all_comments_features.T.to_csv('FinalFeatures_with_comment_time2.csv', encoding='utf-8')


if __name__ == '__main__':
    # only_subreddit_similarity = True if we want to calculate only the similarity between the subreddits,
    # False- if we want all features
    # only_percent = True if we want to calculate only the percent_efficient_references_comment_author,
    # False- if we want all features
    main(only_subreddit_similarity=False, only_percent=False)

