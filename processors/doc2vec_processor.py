import logging

from gensim.models import Doc2Vec
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

from data.labeledline import LabeledLineSentence
from processors.abstract_processor import Processor
from utils import file_helper

log = logging.getLogger(__name__)


class Doc2VecProcessor(Processor):

    def process(self):
        log.info("Doc2VecProcessor begun")

        input_tweets = file_helper.read_input_data(self.options.input_file_path)
        input_tweets_labeled = LabeledLineSentence(self.options.input_file_path)
        model = Doc2Vec(input_tweets_labeled, iter=20, workers=8)

        x_train = list()
        y_train = list()
        count = 1
        for input_tweet in input_tweets:
            log.debug("On tweet " + str(count))
            count += 1
            x_train.append(model.infer_vector(input_tweet.text.split()))
            y_train.append(input_tweet.intensity)

        log.debug("Computing model")
        scores = \
            model_selection.cross_val_score(
                LinearRegression(), x_train, y_train, cv=10, scoring='r2'
            )
        mean_score = scores.mean()

        log.info("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, scores.std() * 2))

        log.info("Doc2VecProcessor ended")
