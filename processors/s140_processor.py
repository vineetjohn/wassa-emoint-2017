import logging

from sklearn import model_selection
from sklearn.svm import LinearSVR

from processors.abstract_processor import Processor
from utils import file_helper

log = logging.getLogger(__name__)


class S140Processor(Processor):

    def process(self):
        log.info("S140Processor begun")

        train_tweets = file_helper.read_input_data(self.options.training_data_file_path)
        s140_lexicon = list(file_helper.read_s140_lexicon(self.options.lexicon_file_path))

        y_train = list()
        x_train = list()

        for tweet in train_tweets:
            x_vector = list()
            for lexicon_word in s140_lexicon:
                if lexicon_word.word in tweet.text:
                    x_vector.append(lexicon_word.score)
                else:
                    x_vector.append(0.0)
            x_train.append(x_vector)
            y_train.append(tweet.intensity)

        scores = \
            model_selection.cross_val_score(
                LinearSVR(), x_train, y_train, cv=10, scoring='r2'
            )
        mean_score = scores.mean()

        log.info("Accuracy: %0.2f (+/- %0.2f)" % (mean_score, scores.std() * 2))

        if self.options.test_data_file_path:

            log.info("Making predictions for the test dataset")

            test_tweets = list(file_helper.read_test_data(self.options.test_data_file_path))

            ml_model = LinearSVR()
            ml_model.fit(x_train, y_train)

            x_test = list()
            for tweet in test_tweets:
                x_vector = list()
                for lexicon_word in s140_lexicon:
                    if lexicon_word.word in tweet.text:
                        x_vector.append(lexicon_word.score)
                    else:
                        x_vector.append(0.0)
                x_test.append(x_vector)
            y_test = ml_model.predict(X=x_test)

            with open(self.options.test_data_file_path + ".predicted", 'w') as predictions_file:
                for i in range(len(test_tweets)):
                    predictions_file.write(
                        str(test_tweets[i].id) + "\t" + test_tweets[i].text + "\t" +
                        test_tweets[i].emotion +"\t" + str(y_test[i]) + "\n"
                    )

        log.info("S140Processor ended")
