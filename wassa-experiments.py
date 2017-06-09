import scipy.stats
import numpy as np
import re
import html
import time
import pickle

import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords

from xgboost import XGBRegressor


word_vector_path = "/home/v2john/"
wassa_home = "/home/v2john/WASSA-Task/"


def evaluate(pred, gold):

    f = open(pred, "rb")
    pred_lines = f.readlines()
    f.close()

    f = open(gold, "rb")
    gold_lines = f.readlines()
    f.close()

    if(len(pred_lines) == len(gold_lines)):
        # align tweets ids with gold scores and predictions
        data_dic = {}

        for line in gold_lines:
            line = line.decode()
            parts = line.split('\t')
            if len(parts) == 4:
                data_dic[int(parts[0])] = [float(line.split('\t')[3])]
            else:
                raise ValueError('Format problem.')

        for line in pred_lines:
            line = line.decode()
            parts = line.split('\t')
            if len(parts) == 4:
                if int(parts[0]) in data_dic:
                    try:
                        data_dic[int(parts[0])].append(float(line.split('\t')[3]))
                    except ValueError:
                        # Invalid predictions are replaced by a default value
                        data_dic[int(parts[0])].append(0.5)
                else:
                    raise ValueError('Invalid tweet id.')
            else:
                raise ValueError('Format problem.')

        # lists storing gold and prediction scores
        gold_scores = []
        pred_scores = []

        # lists storing gold and prediction scores where gold score >= 0.5
        gold_scores_range_05_1 = []
        pred_scores_range_05_1 = []

        for id in data_dic:
            if(len(data_dic[id]) == 2):
                gold_scores.append(data_dic[id][0])
                pred_scores.append(data_dic[id][1])
                if(data_dic[id][0] >= 0.5):
                    gold_scores_range_05_1.append(data_dic[id][0])
                    pred_scores_range_05_1.append(data_dic[id][1])
            else:
                raise ValueError('Repeated id in test data.')

        # return zero correlation if predictions are constant
        if np.std(pred_scores) == 0 or np.std(gold_scores) == 0:
            return (0, 0, 0, 0)

        pears_corr = scipy.stats.pearsonr(pred_scores, gold_scores)[0]
        spear_corr = scipy.stats.spearmanr(pred_scores, gold_scores)[0]

        pears_corr_range_05_1 = scipy.stats.pearsonr(pred_scores_range_05_1, gold_scores_range_05_1)[0]
        spear_corr_range_05_1 = scipy.stats.spearmanr(pred_scores_range_05_1, gold_scores_range_05_1)[0]

        return (pears_corr, spear_corr, pears_corr_range_05_1, spear_corr_range_05_1)
    else:
        raise ValueError('Predictions and gold data have different number of lines.')


def evaluate_lists(pred, gold):
    if len(pred) == len(gold):
        gold_scores = gold
        pred_scores = pred

        # lists storing gold and prediction scores where gold score >= 0.5
        gold_scores_range_05_1 = []
        pred_scores_range_05_1 = []

        for i in range(len(gold_scores)):
            if(gold_scores[i] >= 0.5):
                gold_scores_range_05_1.append(gold_scores[i])
                pred_scores_range_05_1.append(pred_scores[i])

        # return zero correlation if predictions are constant
        if np.std(pred_scores) == 0 or np.std(gold_scores) == 0:
            return (0, 0, 0, 0)

        pears_corr = scipy.stats.pearsonr(pred_scores, gold_scores)[0]
        spear_corr = scipy.stats.spearmanr(pred_scores, gold_scores)[0]

        pears_corr_range_05_1 = scipy.stats.pearsonr(pred_scores_range_05_1, gold_scores_range_05_1)[0]
        spear_corr_range_05_1 = scipy.stats.spearmanr(pred_scores_range_05_1, gold_scores_range_05_1)[0]

        return np.array([pears_corr, spear_corr, pears_corr_range_05_1, spear_corr_range_05_1])
    else:
        raise ValueError('Predictions and gold data have different number of lines.')


# In[5]:
def remove_stopwords(string):
    split_string = \
        [word for word in string.split()
         if word not in stopwords.words('english')]

    return " ".join(split_string)


def clean_str(string):
    string = html.unescape(string)
    string = string.replace("\\n", " ")
    string = string.replace("_NEG", "")
    string = string.replace("_NEGFIRST", "")
    string = re.sub(r"@[A-Za-z0-9_s(),!?\'\`]+", "", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\*", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " ,", string)
    string = re.sub(r"!", " !", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ?", string)
    string = re.sub(r"\s{2,}", " ", string)

    return remove_stopwords(string.strip().lower())


# # Metadata and Class Definitions

class Tweet(object):

    def __init__(self, id, text, emotion, intensity):
        self.id = id
        self.text = text
        self.emotion = emotion
        self.intensity = intensity

    def __repr__(self):
        return \
            "id: " + self.id + \
            ", text: " + self.text + \
            ", emotion: " + self.emotion + \
            ", intensity: " + self.intensity

# In[9]:


def read_training_data(training_data_file_path):

    train_list = list()
    with open(training_data_file_path) as input_file:
        for line in input_file:
            line = line.strip()
            array = line.split('\t')
            train_list.append(Tweet(array[0], clean_str(array[1]), array[2], float(array[3])))
    return train_list


num_test = 65535


def is_active_vector_method(string):
    return int(string)


def vectorize_tweets(tweet_list, bin_string, vector_dict):

    vectors = list()
    frames = list()

    '''Pre-trained Word embeddings'''
    index = 0
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = \
                DataFrame(list(map(lambda x: get_word2vec_embedding(x, wv_model, w2v_dimensions), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 1
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = \
                DataFrame(list(map(lambda x: get_word2vec_embedding(x, wv_model_1, w2v_dimensions_1), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 2
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = \
                DataFrame(list(map(lambda x: get_word2vec_embedding(x, wv_model_2, w2v_dimensions_2), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    '''NRC Emotion Intensity Lexicon'''
    index = 3
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_emo_int_vector(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    '''WordNet'''
    index = 4
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_sentiwordnetscore(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    '''NRC Sentiment Lexica'''
    index = 5
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_sentiment_emotion_feature(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 6
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_sentiment_emoticon_lexicon_vector(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 7
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_sentiment_emoticon_afflex_vector(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    '''NRC Hashtag Lexica'''
    index = 8
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_hashtag_emotion_vector(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 9
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_sentiment_hash_sent_lex_vector(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 10
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_sentiment_hashtag_affneglex_vector(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 11
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = \
                DataFrame(list(map(lambda x: get_word2vec_embedding(x, wv_model_3, w2v_dimensions_3), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 12
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = \
                DataFrame(list(map(lambda x: get_word2vec_embedding(x, wv_model_4, w2v_dimensions_4), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 13
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = \
                DataFrame(list(map(lambda x: get_word2vec_embedding(x, wv_model_5, w2v_dimensions_5), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 14
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_emoji_intensity(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    index = 15
    if is_active_vector_method(bin_string[index]):
        if index not in vector_dict.keys():
            tmp_vector = DataFrame(list(map(lambda x: get_depeche_mood_vector(x), tweet_list)))
            vector_dict[index] = tmp_vector
        frames.append(vector_dict[index])

    vectors = pd.concat(frames, axis=1)

    return vectors.values.tolist()


def restore_vectors(vectors_path):
    vector_dict = None
    with open(vectors_path, 'rb') as vectors_file:
        vector_dict = pickle.load(vectors_file)
    return vector_dict


# In[227]:
def run_test(x_train, score_train, x_test, y_gold):
    ml_model = XGBRegressor(seed=0)

    x_train = np.array(x_train)
    score_train = np.array(score_train)

    ml_model.fit(x_train, score_train)
    y_pred = ml_model.predict(x_test)

    score = evaluate_lists(y_pred, y_gold)

    return score


# In[228]:

def load_all_data(emotion):
    training_data_file_path = \
        wassa_home + "dataset/" + emotion + "-ratings-0to1.train.txt"
    dev_set_path = \
        wassa_home + "dataset/dev-set/" + emotion + "-ratings-0to1.dev.gold.txt"
    test_data_file_path = \
        wassa_home + "dataset/test-set/" + emotion + "-ratings-0to1.test.gold.txt"

    tweet_train = list()
    score_train = list()
    tweet_test = list()
    y_gold = list()

    training_tweets = read_training_data(training_data_file_path)
    for tweet in training_tweets:
        tweet_train.append(tweet.text)
        score_train.append(float(tweet.intensity))

    dev_tweets = read_training_data(dev_set_path)
    for tweet in dev_tweets:
        tweet_train.append(tweet.text)
        score_train.append(float(tweet.intensity))

    test_tweets = read_training_data(test_data_file_path)
    for tweet in test_tweets:
        tweet_test.append(tweet.text)
        y_gold.append(float(tweet.intensity))

    return tweet_train, tweet_test, score_train, y_gold


for emotion in ['anger', 'fear', 'joy', 'sadness']:

    print("Working on: " + emotion)
    tweet_train, tweet_test, score_train, y_gold = load_all_data(emotion)

    result_file_path = "/home/v2john/" + emotion + "_tests.tsv"
    with open(result_file_path, 'a+') as result_file:
        result_file.write(
            "Feature Selection String\t" +
            "Num. Features\t" +
            "Pearson\t" +
            "Spearman\t" +
            "Pearson (0.5-1)\t" +
            "Spearman (0.5-1)\n"
        )

    train_vectors_path = "/home/v2john/" + emotion + "_train_vectors"
    test_vectors_path = "/home/v2john/" + emotion + "_test_vectors"

    train_vector_dict = restore_vectors(train_vectors_path)
    test_vector_dict = restore_vectors(test_vectors_path)

    for i in range(num_test, 0, -1):
        print("Current test: " + str(i) + "/" + str(num_test))
        bin_string = '{0:016b}'.format(i)
        start_time = time.time()

        print("Vectorizing data")
        x_train = vectorize_tweets(tweet_train, bin_string, train_vector_dict)
        x_test = vectorize_tweets(tweet_test, bin_string, test_vector_dict)

        print("Training and testing models")
        train_scores = run_test(x_train, score_train, x_test, y_gold)

        with open(result_file_path, 'a+') as result_file:
            result_file.write(
                "~" + bin_string + "\t" +
                str(len(x_train[0])) + "\t" +
                str(train_scores[0]) + "\t" +
                str(train_scores[1]) + "\t" +
                str(train_scores[2]) + "\t" +
                str(train_scores[3]) + "\n"
            )
        print("--- %s seconds ---" % (time.time() - start_time))
