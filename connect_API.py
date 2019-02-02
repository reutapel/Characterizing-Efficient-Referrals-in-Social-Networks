# import requests
# import json
import praw
import time
import csv


class API_connection:

    def __init__(self, sub_reddit):
        user_agent = "learning API 1.0"
        self.r_connection = praw.Reddit(user_agent=user_agent)
        self.sub_reddit = sub_reddit
        with open('subreddits/subreddit1.csv', 'r') as csvfile:
            self.subreddits_list = list(csv.reader(csvfile))
        csvfile.close()

        with open('subreddits/subreddit2.csv', 'r') as csvfile:
            self.subreddits_list += list(csv.reader(csvfile))
            # self.subreddits_list = self.subreddits_list[:0]
        csvfile.close()

        with open('subreddits/subreddit3.csv', 'r') as csvfile:
            self.subreddits_list += list(csv.reader(csvfile))
        csvfile.close()

        with open('subreddits/subreddit4.csv', 'r') as csvfile:
            self.subreddits_list += list(csv.reader(csvfile))
        csvfile.close()

        return

    def subreddit(self, post_limit):
        subreddit = self.r_connection.get_subreddit(self.sub_reddit)
        subreddit_posts = subreddit.get_hot(limit=post_limit)
        subids = set()
        for submission in subreddit_posts:
            subids.add(submission.id)
        subid = list(subids)
        return subid

    def parse_comments(self, post_limit, subid, string_to_find):
        reference_comments = []
        index = min(post_limit, len(subid))
        for i in range(0, index):
            print('{}: start submission {}'.format((time.asctime(time.localtime(time.time()))), i))
            submission = self.r_connection.get_submission(submission_id=subid[i])
            print('{}: start more comments'.format((time.asctime(time.localtime(time.time())))))
            submission.replace_more_comments(limit=None, threshold=0)
            print('{}: start flat comments'.format((time.asctime(time.localtime(time.time())))))
            flat_comments = praw.helpers.flatten_tree(submission.comments)
            # with open('alreadydone.txt', 'r') as f:
            #     already_done = [line.strip() for line in f]
            # f.close()
            for comment in flat_comments:
                # if comment.id==u'da7f3xh':
                #     r=1
                if string_to_find in comment.body:
                    reference_subreddit = comment.body[comment.body.find(string_to_find):].split(('/')) #split the body from /r/
                    reference_subreddit = reference_subreddit[2][:reference_subreddit[2].find(' ')].encode('utf-8') #find the subreddit name in the reference
                    reference_subreddit = [reference_subreddit]
                    if 'edit' not in comment.body:
                        if 'I am a bot, and this action was performed automatically' not in comment.body:
                            if string_to_find+self.sub_reddit not in comment.body:
                                if reference_subreddit in self.subreddits_list:
                                    reference_comments.append(comment)

            # real_comments = [comment for comment in flat_comments if string_to_find in comment.body]
            # reference_comments += real_comments
        print(len(reference_comments))
        with open(self.sub_reddit + ' reference comments.csv', 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            fieldnames2 = ['comment_body', 'comment_path', 'comment_id', 'parent_id', 'submission_id']
            writer.writerow(fieldnames2)
            # for comment in reference_comments:
            writer.writerow([comment.body.encode('utf-8'), comment.permalink.encode('utf-8'),
                                 comment.id.encode('utf-8'), comment.parent_id.encode('utf-8'), comment.submission.id.encode('utf-8')])

        return


def main():
    post_limit = 3000
    string_to_find = '/r/'
    sub_reddit = "diet"
    print('Run with {} hot submissions for sub reddit {}'.format(post_limit, sub_reddit))
    connect = API_connection(sub_reddit)
    subid = connect.subreddit(post_limit)
    connect.parse_comments(post_limit, subid, string_to_find)


if __name__ == '__main__':
    main()