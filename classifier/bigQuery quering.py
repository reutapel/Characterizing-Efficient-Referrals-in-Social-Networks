import os
import argparse
import logging
# from googleapiclient.discovery import build
# from apiclient.discovery import build
# from apiclient.errors import HttpError
# from oauth2client.client import GoogleCredentials
# from google.cloud import bigquery
from pandas.io import gbq
import pandas as pd
import time
import csv
import Train_test_classifier

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'C:\\Users\\user\\Documents\\Technion\\semester 8\\My Project-b7f535d58ff7.json'
MyProjectID = 'add one'


def FirstQuery():
    # [START run_query]
    subreddits_list = ('diet', '1200isplenty', 'bjj', 'EatCheapAndHealthy',
                  'fasting', 'gainit', 'intermittentfasting', 'leangains', 'loseit', 'nutrition',
                  'PlantBasedDiet', '100DaysofKeto', 'ketogains', 'keto', 'ketoscience', 'vegetarianketo', 'xxketo',
                   '90daysgoal', 'MealPrepSunday', 'zerocarb', 'veganketo', 'fatlogic', 'trueloseit', 'ketorecipes',
                   'ketochow', 'intermittentfasting', 'ketoaustralia', 'ketodrunk', 'ketonz', 'vegan1200isplenty',
                   'CuttingWeight', 'ketotrees', 'sugarfree', 'DesiKeto', 'ketouk', 'ketofreestyle', '1200isjerky',
                   'ketocirclejerk', 'askfatlogic', 'FoodAddiction', 'dutchketo', 'fatacceptance', 'Diets', 'Keto_Food')
    print('subreddits_list length is: {}'.format(len(subreddits_list)))

    # subreddits_list = ('diet', 'paleo')  #small subreddits for tests

    print('{}: Start comments queries'.format((time.asctime(time.localtime(time.time())))))
    logging.debug('Start comments queries')

    # Query the comments tables in bugquerry and import the results
    commentDF = pd.DataFrame()
    for subreddit in subreddits_list:
        comment_query = """SELECT body AS comment_body, author AS comment_author, created_utc AS comment_created_time,
                            subreddit_id, link_id, parent_id, id AS comment_id, subreddit FROM
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_01]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_02]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_03]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_04]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_05]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_06]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_07]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_08]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_09]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_10]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_11]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_12]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_01]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_02]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_03]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_04]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_05]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_06]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_07]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_08]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_09]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_10]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_11]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_12]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2017_01]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2017_02])
                            WHERE subreddit = '{}' AND body LIKE '%/r/%';""" .format(subreddit)
        print('{}: Start quering the query: {}'. \
            format((time.asctime(time.localtime(time.time()))), comment_query))
        logging.debug('Start quering the query: {}'.format(comment_query))
        commentDF = commentDF.append(gbq.read_gbq(comment_query, project_id=MyProjectID))

    print('{}: Finish comments queries, start submissions queries'. \
        format((time.asctime(time.localtime(time.time())))))
    logging.debug('Finish comments queries, start submissions queries')

    # Query the posts tables in bugquerry and import the results
    submissionsDF = pd.DataFrame()
    for subreddit in subreddits_list:
        submissions_query = """SELECT created_utc AS submission_created_time, subreddit AS IsEfficient, url, num_comments, title,
                            selftext AS submission_body, author AS submission_author, id AS submission_id,
                            subreddit_id AS recommend_subreddit, name AS link_id FROM
                            (SELECT * FROM [fh-bigquery:reddit_posts.2015_12]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_01]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_02]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_03]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_04]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_05]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_06]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_07]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_08]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_09]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_10]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_11]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_12]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2017_01]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2017_02])
                            WHERE subreddit = '{}';""" .format(subreddit)
        print('{}: Start quering the query: {}'. \
            format((time.asctime(time.localtime(time.time()))), submissions_query))
        logging.debug('Start quering the query: {}'.format(submissions_query))
        submissionsDF = submissionsDF.append(gbq.read_gbq(submissions_query, project_id=MyProjectID))

    # merge (join) the results from posts and comments tables.
    merge_results = pd.merge(commentDF, submissionsDF, on='link_id')
    return merge_results


#first filter of the comments- check that the subreddits in the comments are ok
def filter_results(TestDate, string_to_find):
    reference_comments = pd.DataFrame()

    #upload the names of the exists subreddits
    with open('C:/gitprojects/final project/subreddits/subreddit1.csv', 'r') as csvfile:
        subreddits_list = list(csv.reader(csvfile))
    csvfile.close()

    with open('C:/gitprojects/final project/subreddits/subreddit2.csv', 'r') as csvfile:
        subreddits_list += list(csv.reader(csvfile))
    csvfile.close()

    with open('C:/gitprojects/final project/subreddits/subreddit3.csv', 'r') as csvfile:
        subreddits_list += list(csv.reader(csvfile))
    csvfile.close()

    with open('C:/gitprojects/final project/subreddits/subreddit4.csv', 'r') as csvfile:
        subreddits_list += list(csv.reader(csvfile))
    csvfile.close()

    # for each comment, find the subreddit in it, and check that it is not the same as the comment's subreddit and that
    # there is such subreddit.
    counter_r = 0
    for index, comment in TestDate.iterrows():
        comment_body = comment['comment_body'].encode('utf-8')
        reference_subreddit = comment_body[comment_body.find(string_to_find):].split(('/'))  # split the body from /r/
        number_of_r = 0
        for i in range(0, len(reference_subreddit)):
            if reference_subreddit[i] == 'r':
                number_of_r += 1
        if '\n' in reference_subreddit[2]:
            reference_subreddit[2] = reference_subreddit[2].split(('\n'))
            reference_subreddit[2] = reference_subreddit[2][0]
        if ' ' in reference_subreddit[2]:
            reference_subreddit = [reference_subreddit[2][:reference_subreddit[2].find(' ')]]  # find the subreddit name in the reference
        if len(reference_subreddit) >= 2:
            if '!' in reference_subreddit[2]:
                reference_subreddit = [reference_subreddit[2][:reference_subreddit[2].find('!')]]  # find the subreddit name in the reference
            elif ')' in reference_subreddit[2]:
                reference_subreddit = [reference_subreddit[2][:reference_subreddit[2].find(')')]]  # find the subreddit name in the reference
            elif ']' in reference_subreddit[2]:
                reference_subreddit = [reference_subreddit[2][:reference_subreddit[2].find(']')]]  # find the subreddit name in the reference
            elif '.' in reference_subreddit[2]:
                reference_subreddit = [reference_subreddit[2][:reference_subreddit[2].find('.')]]  # find the subreddit name in the reference
            elif '?' in reference_subreddit[2]:
                reference_subreddit = [reference_subreddit[2][:reference_subreddit[2].find('?')]]  # find the subreddit name in the reference
            elif ',' in reference_subreddit[2]:
                reference_subreddit = [reference_subreddit[2][:reference_subreddit[2].find(',')]]  # find the subreddit name in the reference
            elif '"' in reference_subreddit[2]:
                reference_subreddit = [reference_subreddit[2][:reference_subreddit[2].find('"')]]  # find the subreddit name in the reference
            else:
                reference_subreddit = [reference_subreddit[2]]
        else:
            if '!' in reference_subreddit[0]:
                reference_subreddit = [reference_subreddit[0][:reference_subreddit[0].find('!')]]  # find the subreddit name in the reference
            if ')' in reference_subreddit[0]:
                reference_subreddit = [reference_subreddit[0][:reference_subreddit[0].find(')')]]  # find the subreddit name in the reference
            if ']' in reference_subreddit[0]:
                reference_subreddit = [reference_subreddit[0][:reference_subreddit[0].find(']')]]  # find the subreddit name in the reference
            if '.' in reference_subreddit[0]:
                reference_subreddit = [reference_subreddit[0][:reference_subreddit[0].find('.')]]  # find the subreddit name in the reference
            if '?' in reference_subreddit[0]:
                reference_subreddit = [reference_subreddit[0][:reference_subreddit[0].find('?')]]  # find the subreddit name in the reference
            if ',' in reference_subreddit[0]:
                reference_subreddit = [reference_subreddit[0][:reference_subreddit[0].find(',')]]  # find the subreddit name in the reference
            if '"' in reference_subreddit[0]:
                reference_subreddit = [reference_subreddit[0][:reference_subreddit[0].find('"')]]  # find the subreddit name in the reference
            else:
                reference_subreddit = [reference_subreddit[0]]
        if comment['submission_author'] != '[deleted]':
            if reference_subreddit[0] != comment['subreddit'].encode('utf-8'):
                if 'edit' not in comment_body:
                    if 'I am a bot, and this action was performed automatically' not in comment_body:
                        if reference_subreddit in subreddits_list:
                            comment['recommend_subreddit'] = reference_subreddit[0]
                            reference_comments = reference_comments.append(comment)
                            if number_of_r > 1:
                                counter_r += 1
    print('number of comments with more than 1 reference is: {}'.format(counter_r))

    return reference_comments


def SecondQuery(predictedResults):
    predictedResultsEfficiency = pd.DataFrame()
    # predictedResultsEfficiency = pd.read_csv('FinalResultsWithEfficient_13.4.17_7.csv')

    for index, comment in predictedResults.iterrows():
        if comment['classifier_result'] > 0.9: # and comment['classifier_result'] < 0.985890227:
            # query: check if the author of the submission posted in the recommended subreddit before the reference
            is_post_before_query = """SELECT * FROM
                            (SELECT * FROM [fh-bigquery:reddit_posts.2015_12]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_01]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_02]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_03]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_04]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_05]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_06]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_07]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_08]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_09]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_10]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_11]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2016_12]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2017_01]),
                            (SELECT * FROM [fh-bigquery:reddit_posts.2017_02]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_01]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_02]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_03]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_04]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_05]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_06]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_07]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_08]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_09]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_10]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_11]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2015_12]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_01]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_02]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_03]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_04]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_05]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_06]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_07]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_08]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_09]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_10]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_11]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2016_12]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2017_01]),
                            (SELECT * FROM [fh-bigquery:reddit_comments.2017_02])
                     WHERE author = '{}'
                     AND created_utc < {}
                     AND subreddit = '{}'""".format(comment['submission_author'], comment['comment_created_time'],
                                                  comment['recommend_subreddit'])
            # print '{}: Start quering the query: author: {}, create_utd: {}, recommended_subreddit:{}'. \
            #     format((time.asctime(time.localtime(time.time()))), comment['submission_author'],
            #            comment['comment_created_time'], comment['recommend_subreddit'])
            print('{}: Start quering the query: {} '.\
                format((time.asctime(time.localtime(time.time()))), is_post_before_query))
            logging.debug('{}: Start quering the query: {}'.\
                          format((time.asctime(time.localtime(time.time()))), is_post_before_query))
            is_post_before_df = (gbq.read_gbq(is_post_before_query, project_id=MyProjectID))

            if is_post_before_df.empty:
                #query: check if the author of the submission posted in the recommended subreddit after the reference
                is_post_after_query = """SELECT * FROM
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2015_12]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_01]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_02]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_03]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_04]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_05]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_06]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_07]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_08]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_09]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_10]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_11]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2016_12]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2017_01]),
                                    (SELECT * FROM [fh-bigquery:reddit_posts.2017_02]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_01]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_02]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_03]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_04]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_05]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_06]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_07]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_08]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_09]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_10]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_11]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2015_12]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_01]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_02]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_03]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_04]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_05]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_06]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_07]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_08]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_09]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_10]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_11]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2016_12]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2017_01]),
                                    (SELECT * FROM [fh-bigquery:reddit_comments.2017_02])
                             WHERE author = '{}'
                             AND created_utc > {}
                             AND subreddit = '{}'""".format(comment['submission_author'], comment['comment_created_time'],
                                                          comment['recommend_subreddit'])
                # print '{}: Start quering the query: author: {}, create_utd: {}, recommended_subreddit:{}'. \
                #     format((time.asctime(time.localtime(time.time()))), comment['submission_author'],
                #            comment['comment_created_time'], comment['recommend_subreddit'])
                print('{}: Start quering the query: {} '. \
                    format((time.asctime(time.localtime(time.time()))), is_post_after_query))
                logging.debug('{}: Start quering the query: {}'.\
                              format((time.asctime(time.localtime(time.time()))), is_post_after_query))
                is_post_after_df = (gbq.read_gbq(is_post_after_query, project_id=MyProjectID))
                if is_post_after_df.empty:
                    # the author of the submission posted in the recommended subreddit after the reference but not before
                    comment['IsEfficient'] = -1
                    predictedResultsEfficiency = predictedResultsEfficiency.append(comment)
                else: # the author of the submission didn't post in the recommended subreddit after the reference and not before
                    comment['IsEfficient'] = 1
                    predictedResultsEfficiency = predictedResultsEfficiency.append(comment)
            else: # the author of the submission posted in the recommended subreddit after the reference and also before
                comment['IsEfficient'] = -1
                predictedResultsEfficiency = predictedResultsEfficiency.append(comment)
            predictedResultsEfficiency.to_csv('FinalResultsWithEfficientSub.csv', encoding='utf-8')

    return predictedResultsEfficiency


if __name__ == '__main__':
    logging.basicConfig(filename='logfile_{}.log'.format(time.localtime()), level=logging.DEBUG)
    print('{}: Start quering'.format((time.asctime(time.localtime(time.time())))))
    logging.info('{}: Start quering'.format((time.asctime(time.localtime(time.time())))))
    merge_results = FirstQuery()
    merge_results.to_csv('before_filtering_1447.csv', encoding='utf-8')
    print('{}: Initial filtering'.format((time.asctime(time.localtime(time.time())))))
    logging.debug('{}: Initial filtering'.format((time.asctime(time.localtime(time.time())))))
    # merge_results = pd.read_csv('before_filtering_1447.csv')
    filtered_results = filter_results(merge_results, '/r/')
    filtered_results.to_csv('filter_results_1447.csv', encoding='utf-8')
    # filtered_results = pd.read_csv('filter_results_1447.csv')
    test_data = filtered_results[['comment_body', 'comment_id', 'submission_id']]
    # test_data = filtered_results.loc[: 'comment_body']
    print('{}: Number of samples is: {}'.format((time.asctime(time.localtime(time.time()))), test_data.shape[0]))
    logging.debug('{}: Number of samples is: {}'.format((time.asctime(time.localtime(time.time()))), test_data.shape[0]))
    predicted = Train_test_classifier.main(test_data)
    # predictedPD = pd.DataFrame(predicted[:, 1])
    # returnPredicted = pd.concat([filtered_results, predictedPD], axis=1)
    resultsPredicted = pd.merge(filtered_results, predicted, on=('comment_body', 'comment_id', 'submission_id'))
    resultsPredicted.to_csv('resultsPredicted1600.csv', encoding='utf-8')
    logging.debug('{}: Start'.format((time.asctime(time.localtime(time.time())))))
    resultsPredicted = pd.read_csv('resultsPredicted1600.csv')
    resultsPredictedSort = resultsPredicted.sort(columns=['classifier_result'], ascending=False, axis=0)
    FinalResultsWithEfficient = SecondQuery(resultsPredictedSort)
    FinalResultsWithEfficient.to_csv('FinalResultsWithEfficient_final.csv', encoding='utf-8')
# [END all]