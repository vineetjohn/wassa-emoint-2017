import logging

from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVR

from processors.abstract_processor import Processor
from utils import file_helper

log = logging.getLogger(__name__)


class NGramProcessor(Processor):

    def process(self):
        log.info("NGramProcessor begun")

        input_tweets = file_helper.read_input_data(self.options.training_data_file_path)

        y_train = list()
        tweet_train = list()
        for input_tweet in input_tweets:
            tweet_train.append(input_tweet.text)
            y_train.append(input_tweet.intensity)

        vectorizer = CountVectorizer(ngram_range=(1, 2))
        x_train = vectorizer.fit_transform(tweet_train)

        scores = \
            model_selection.cross_val_score(
                LinearSVR(), x_train, y_train, cv=10, scoring='r2'
            )
        mean_score = scores.mean()

        log.info("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, scores.std() * 2))

        if self.options.test_data_file_path:

            log.info("Making predictions for the test dataset")

            test_tweets = list(file_helper.read_test_data(self.options.test_data_file_path))
            tweet_test = list()
            for tweet in test_tweets:
                tweet_test.append(tweet.text)

            tweet_train.extend(tweet_test)

            x_train = vectorizer.fit_transform(tweet_train)[:len(y_train)]
            x_test = vectorizer.fit_transform(tweet_train)[len(y_train):]

            ml_model = LinearSVR()
            ml_model.fit(x_train, y_train)

            y_test = ml_model.predict(X=x_test)
            with open(self.options.test_data_file_path + ".predicted", 'w') as predictions_file:
                for i in range(len(y_test)):
                    predictions_file.write(
                        str(test_tweets[i].id) + "\t" + test_tweets[i].text + "\t" +
                        test_tweets[i].emotion +"\t" + str(y_test[i]) + "\n"
                    )

        log.info("NGramProcessor ended")
