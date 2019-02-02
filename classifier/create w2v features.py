from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from datetime import datetime
import time
from sklearn.preprocessing import scale, MinMaxScaler
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


# Create log file
LOG_FILENAME = datetime.now().strftime('CreateW2V_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO,)


class TextClassifier:
    def __init__(self):
        print('{}: Loading the data '.format((time.asctime(time.localtime(time.time())))))
        sentences = []
        labels = []
        sentences_len = []
        stripComment = lambda x: x.strip().lower()
        replaceComments = lambda x: x.replace(";", ' ').replace(":", ' ').replace('"', ' ').replace('-', ' ').\
            replace(',', ' ').replace('.', ' ').replace("/", ' ').replace('(', ' ').replace(')', ' ')
        splitCommant = lambda x: x.split(" ")
        stop = stopwords.words('english')
        stopWordsComment = lambda x: [i for i in x if i not in stop]
        self.data = pd.read_excel('Features_causality_final_no_delete_author.xlsx')
        # parse the comments to vectors of words:
        for index, comment in self.data.iterrows():
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

        # create data
        self.X_train, self.labels = sentences, labels
        comments_id = self.data['comment_id'].values
        self.comment_id = comments_id.tolist()  # list of all comment IDs
        # Create w2v model
        print('{}: Create w2v'.format((time.asctime(time.localtime(time.time())))))

        sentence_maxlen = 100
        w2v = Word2Vec(size=sentence_maxlen)
        token_list = [x for x in tqdm(self.X_train)]
        w2v.build_vocab(token_list)
        w2v.train(token_list, total_examples=len(self.X_train), epochs=w2v.epochs)

        print('{}: building tf-idf matrix ... '.format((time.asctime(time.localtime(time.time())))))
        vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
        vectorizer.fit_transform([x for x in self.X_train])
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        print('{}: vocab size {} '.format((time.asctime(time.localtime(time.time()))), len(tfidf)))

        train_vecs_w2v = np.concatenate([self.buildWordVector(z, sentence_maxlen, tfidf, w2v)
                                         for z in tqdm(map(lambda x: x, self.X_train))])

        # Normalized the w2v vector
        train_vecs_w2v = scale(train_vecs_w2v)
        # for Non-Negative values - if we want to train Multinumial NB
        min_max_scale = MinMaxScaler()
        train_vecs_w2v = min_max_scale.fit_transform(train_vecs_w2v)

        i = 0
        w2v_id = []
        for sample in train_vecs_w2v:
            w2v_id_sample = sample.tolist()
            w2v_id_sample.append(comments_id[i])
            w2v_id.append(w2v_id_sample)
            i += 1

        index = list(range(sentence_maxlen))
        index.append('comment_id')
        train_vecs_w2vPD = pd.DataFrame.from_records(w2v_id, columns=index)
        train_vecs_w2vPD.to_csv('train_vecs_w2vPD_causality.csv', encoding='utf-8')
        final_features = pd.merge(self.data, train_vecs_w2vPD, on='comment_id')
        final_features.to_csv('100w2v_scale_2_causality.csv', encoding='utf-8')

        return

    def buildWordVector(self, tokens, size, tfidf, w2v):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tokens:
            try:
                vec += w2v[word].reshape((1, size)) * tfidf[word]
                count += 1.
            except KeyError:  # handling the case where the token is not
                # in the corpus. useful for testing.
                continue
        if count != 0:
            vec /= count
        return vec


#######################################################################################################
if __name__ == '__main__':
    text_classifier = TextClassifier()
