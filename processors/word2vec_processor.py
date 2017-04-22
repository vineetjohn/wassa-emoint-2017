import logging

import gensim
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

from processors.abstract_processor import Processor
from utils import file_helper

log = logging.getLogger(__name__)


class Word2VecProcessor(Processor):

    def process(self):
        log.info("Word2VecProcessor begun")

        wv_model = \
            gensim.models.Word2Vec.\
            load_word2vec_format(
                self.options.wv_model_path, binary=True
            )
        input_tweets = file_helper.read_input_data(self.options.input_file_path)

        x_train = list()
        y_train = list()
        for input_tweet in input_tweets:
            split_text_list = input_tweet.text.split()

            vector_list = list()
            for word in split_text_list:
                try:
                    vector_list.append(wv_model.wv[word])
                except Exception:
                    pass

            sentence_vector = sum(vector_list) / float(len(vector_list))

            x_train.append(sentence_vector)
            y_train.append(input_tweet.intensity)

        scores = \
            model_selection.cross_val_score(
                LinearSVR(), x_train, y_train, cv=10, scoring='r2'
            )
        mean_score = scores.mean()

        log.info("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, scores.std() * 2))

        log.info("Word2VecProcessor ended")
