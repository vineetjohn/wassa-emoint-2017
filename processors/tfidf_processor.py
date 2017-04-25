import logging

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

from processors.abstract_processor import Processor
from utils import file_helper

log = logging.getLogger(__name__)


class TfIdfProcessor(Processor):

    def process(self):
        log.info("TfIdfProcessor begun")

        input_tweets = file_helper.read_input_data(self.options.input_file_path)

        y_all = list()
        tweet_text = list()
        for input_tweet in input_tweets:
            tweet_text.append(input_tweet.text)
            y_all.append(input_tweet.intensity)

        y_train = y_all[84:]
        # print(y_train)

        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        x_all = vectorizer.fit_transform(tweet_text)
        x_train = x_all[84:]
        y_train = list(map(float, y_train))
        x_test = x_all[:84]

        scores = \
            model_selection.cross_val_score(
                LinearSVR(), x_train, y_train, cv=10, scoring='r2'
            )
        mean_score = scores.mean()
        log.info("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, scores.std() * 2))

        ml_model = LinearSVR()
        ml_model.fit(x_train, y_train)

        y_test = ml_model.predict(X=x_test)

        test_tweets = file_helper.read_test_data("/home/v2john/MEGA/Academic/Masters/UWaterloo/Research/WASSA-Task/dataset/anger-ratings-0to1.dev.target.txt")
        count = 0
        for tweet in test_tweets:
            print(str(tweet.id) + "\t" + tweet.text +
                  "\t" + tweet.emotion + "\t" + str(y_test[count]))
            count += 1

        log.info("TfIdfProcessor ended")
