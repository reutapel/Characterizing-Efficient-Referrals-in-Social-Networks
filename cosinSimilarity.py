import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import math
from collections import Counter


def cosinSimilarity(data_file, sub_reddit):
    with open(data_file, 'r') as csvfile:
        comments_to_find = list(csv.reader(csvfile))
        i = 0
        for row in comments_to_find:
            if row[0] == 'comment_body':
                del comments_to_find[i]
                break
            i += 1
    csvfile.close()

    reference_comments = []
    for row in comments_to_find:
        parent = text_to_vector(row[8])
        comment = text_to_vector(row[0])
        original = text_to_vector(row[7])

        cosine_parent = get_cosine(parent, comment)
        cosine_original = get_cosine(original, comment)
        reference_comments.append([row, cosine_parent, cosine_original])
    # for row in comments_to_find:
    #     parent_comment = [f for f in [row[0], row[8]]]
    #     tfidf_parent = TfidfVectorizer(analyzer='char').fit_transform(parent_comment)
    #     parent_similarity = tfidf_parent * tfidf_parent.T
    #     original_comment = [f for f in [row[0], row[7]]]
    #     tfidf_original = TfidfVectorizer().fit_transform(original_comment)
    #     original_similarity = tfidf_original * tfidf_original.T
    #     reference_comments.append([row, parent_similarity.data[0], original_similarity.data[0]])

    with open(sub_reddit + ' cosin similarity.csv', 'w') as newfile:
        writer = csv.writer(newfile, lineterminator='\n')
        fieldnames2 = ['comment_body', 'comment_path', 'comment_id', 'parent_id', 'submission_id', 'Reut_label',
                        'comments', 'original_message', 'parent_body', 'parent_similarity', 'original_similarity',
                        'parent_Cosinsimilarity', 'original_Cosinsimilarity']
        writer.writerow(fieldnames2)
        for comment in reference_comments:
            writer.writerow(
                [comment[0][0], comment[0][1], comment[0][2], comment[0][3], comment[0][4], comment[0][5], comment[0][6],
                 comment[0][7], comment[0][8], comment[0][9], comment[0][10], comment[0][10], comment[1], comment[2]])
    newfile.close


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


def text_to_vector(text):
    WORD = re.compile(r'\w+')
    words = WORD.findall(text)
    return Counter(words)


if __name__ == '__main__':
    data_file = 'diet original message.csv'
    sub_reddit = 'diet'
    cosinSimilarity(data_file, sub_reddit)