import scipy.stats
import gensim
import numpy as np
import re
import html
import json
import time

from sklearn.preprocessing import PolynomialFeatures

from nltk import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn

import pandas as pd
from pandas import DataFrame

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


# # Load pre-trained word vectors


# Google news pretrained vectors
wv_model_path = word_vector_path + "GoogleNews-vectors-negative300.bin.gz"
print("Loading Google News word2vec model")
wv_model = gensim.models.KeyedVectors.load_word2vec_format(wv_model_path, binary=True, unicode_errors='ignore')


# In[59]:

# Twitter pretrained vectors
wv_model_path_1 = word_vector_path + "word2vec_twitter_model.bin"
print("Loading Twitter word2vec model")
wv_model_1 = gensim.models.KeyedVectors.load_word2vec_format(wv_model_path_1, binary=True, unicode_errors='ignore')


# In[60]:

def loadGloveModel(gloveFile):
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        try:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = np.array(embedding)
        except Exception as e:
            print("Skipping line")
    print("Done.", len(model), " words loaded!")
    return model


wv_model_path_2 = word_vector_path + "glove.twitter.27B.200d.txt"
print("Loading Glove model")
wv_model_2 = loadGloveModel(wv_model_path_2)

wv_model_path_3 = word_vector_path + "glove.6B.300d.txt"
wv_model_3 = loadGloveModel(wv_model_path_3)


# In[16]:

wv_model_path_4 = word_vector_path + "glove.42B.300d.txt"
wv_model_4 = loadGloveModel(wv_model_path_4)


# In[17]:

wv_model_path_5 = word_vector_path + "glove.840B.300d.txt"
wv_model_5 = loadGloveModel(wv_model_path_5)

# In[61]:

w2v_dimensions = len(wv_model['word'])
w2v_dimensions_1 = len(wv_model_1['word'])
w2v_dimensions_2 = len(wv_model_2['word'])
w2v_dimensions_3 = len(wv_model_3['word'])
w2v_dimensions_4 = len(wv_model_4['word'])
w2v_dimensions_5 = len(wv_model_5['word'])
print(w2v_dimensions, w2v_dimensions_1, w2v_dimensions_2)


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

# In[8]:

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


def read_test_data(training_data_file_path):

    test_list = list()
    with open(training_data_file_path) as input_file:
        for line in input_file:
            line = line.strip()
            array = line.split('\t')
            test_list.append(Tweet(array[0], clean_str(array[1]), array[2], None))
    return test_list


# # Feature Extraction Snippets

with open(wassa_home + 'lexicons/emoji_map.json') as emoji_file:
    emoji_list = json.load(emoji_file)

emoji_dict = dict()
for emoji in emoji_list:
    emoji_dict[emoji["emoji"]] = (emoji["name"], emoji["polarity"])

poly_emoji_intensity = PolynomialFeatures(5)


def get_emoji_intensity(tweet):
    score = 0.0
    for emoji in emoji_dict.keys():
        count = tweet.count(emoji)
        score += count * emoji_dict[emoji][1]

    return poly_emoji_intensity.fit_transform(np.array([score]).reshape(1, -1))[0].tolist()


# ## Emotion Intensity Lexicon

# In[92]:

affect_intensity_file_path = \
    wassa_home + \
    "lexicons/NRC-AffectIntensity-Lexicon.txt"


def get_word_affect_intensity_dict(emotion):
    word_intensities = dict()

    with open(affect_intensity_file_path) as affect_intensity_file:
        for line in affect_intensity_file:
            word_int_array = line.replace("\n", "").split("\t")

            if (word_int_array[2] == emotion):
                word_intensities[word_int_array[0]] = float(word_int_array[1])

    return word_intensities


# In[93]:


word_intensities = None


# In[94]:

poly_emo_int = PolynomialFeatures(10)


def get_emo_int_vector(tweet):
    score = 0.0
    for word in word_intensities.keys():
        if word in tweet:
            score += tweet.count(word) * float(word_intensities[word])

    return poly_emo_int.fit_transform(np.array([score]).reshape(1, -1))[0].tolist()


# ## Word2Vec + GloVe

# In[95]:

def get_word2vec_embedding(tweet, model, dimensions):
    vector_list = list()
    for word in tweet.split():
        try:
            vector_list.append(model[word])
        except Exception as e:
            pass

    if len(vector_list) == 0:
        vec_rep = np.zeros(dimensions).tolist()
    else:
        try:
            vec_rep = sum(vector_list) / float(len(vector_list))
        except Exception as e:
            print(vector_list)
            print(e)
            raise Exception

    return vec_rep


# ## SentiWordNet

poly_sentiwordnet = PolynomialFeatures(5)


def get_sentiwordnetscore(tweet):

    tweet_score = np.zeros(2)

    for word in tweet.split():
        synsetlist = list(swn.senti_synsets(word))

        if synsetlist:
            tweet_score[0] += synsetlist[0].pos_score()
            tweet_score[1] += synsetlist[0].neg_score()

    sentiwordnetscore_list = poly_sentiwordnet.fit_transform(tweet_score.reshape(1, -1))[0].tolist()

    return sentiwordnetscore_list


# ## Sentiment Emotion Presence Lexicon

# In[97]:

sentiment_emotion_lex_file_path = \
    wassa_home + \
    "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Emotion-Lexicon-v0.92/" + \
    "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"


def get_affect_presence_list(emotion):
    affect_presence_word_list = list()

    with open(sentiment_emotion_lex_file_path) as sentiment_emotion_lex_file:
        for line in sentiment_emotion_lex_file:
            word_array = line.replace("\n", "").split("\t")

            if (word_array[1] == emotion and word_array[2] == '1'):
                affect_presence_word_list.append(word_array[0])

    return affect_presence_word_list


# In[98]:

affect_presence_word_list = None


def get_sentiment_emotion_feature(tweet):
    for word in affect_presence_word_list:
        if word in tweet.split():
            return [1.0]

    return [0.0]


# ## Hashtag Emotion Intensity

# In[47]:

hashtag_emotion_lex_file_path = \
    wassa_home + \
    "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2/" + \
    "NRC-Hashtag-Emotion-Lexicon-v0.2.txt"


def get_hashtag_emotion_intensity(emotion):
    hastag_intensities = dict()

    with open(hashtag_emotion_lex_file_path) as hashtag_emotion_lex_file:
        for line in hashtag_emotion_lex_file:
            word_array = line.replace("\n", "").split("\t")

            if (word_array[0] == emotion):
                hastag_intensities[clean_str(word_array[1])] = float(word_array[2])

    return hastag_intensities


# In[48]:

hashtag_emotion_intensities = None


# In[49]:

poly_emo_int = PolynomialFeatures(10)


def get_hashtag_emotion_vector(tweet):
    score = 0.0
    for word in hashtag_emotion_intensities.keys():
        if word in tweet:
            score += tweet.count(word) * float(hashtag_emotion_intensities[word])

    return poly_emo_int.fit_transform(np.array([score]).reshape(1, -1))[0].tolist()


# ## Emoticon Sentiment Lexicon

# In[12]:

emoticon_lexicon_unigrams_file_path = \
    wassa_home + \
    "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Emoticon-Lexicon-v1.0/Emoticon-unigrams.txt"
emoticon_lexicon_bigrams_file_path = \
    wassa_home + \
    "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Emoticon-Lexicon-v1.0/Emoticon-bigrams.txt"
emoticon_lexicon_pairs_file_path = \
    wassa_home + \
    "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/NRC-Emoticon-Lexicon-v1.0/Emoticon-pairs.txt"
pair_split_string = "---"

emoticon_lexicon_unigrams = dict()
emoticon_lexicon_bigrams = dict()
emoticon_lexicon_pairs = dict()


def get_emoticon_lexicon_unigram_dict():
    with open(emoticon_lexicon_unigrams_file_path) as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_lexicon_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])

    return emoticon_lexicon_unigrams


def get_emoticon_lexicon_bigram_dict():
    with open(emoticon_lexicon_bigrams_file_path) as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_lexicon_bigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])

    return emoticon_lexicon_bigrams


def get_emoticon_lexicon_pairs_dict():
    with open(emoticon_lexicon_pairs_file_path) as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            pair = word_array[0].split(pair_split_string)
            token_1 = clean_str(pair[0])
            token_2 = clean_str(pair[1])
            if token_1 and token_2:
                token_1_dict = None
                if token_1 in emoticon_lexicon_pairs.keys():
                    token_1_dict = emoticon_lexicon_pairs[token_1]
                else:
                    token_1_dict = dict()

                token_1_dict[token_2] = np.array([float(val) for val in word_array[1:]])
                emoticon_lexicon_pairs[token_1] = token_1_dict

    return emoticon_lexicon_pairs


# In[32]:

print("Loading emoticon_lexicon_unigram_dict")
emoticon_lexicon_unigram_dict = get_emoticon_lexicon_unigram_dict()

print("Loading emoticon_lexicon_bigram_dict")
emoticon_lexicon_bigram_dict = get_emoticon_lexicon_bigram_dict()

print("Loading emoticon_lexicon_pairs_dict")
emoticon_lexicon_pairs_dict = get_emoticon_lexicon_pairs_dict()


poly_emoticon_lexicon = PolynomialFeatures(5)


def get_unigram_sentiment_emoticon_lexicon_vector(tokens):
    vector_list = np.zeros(3)
    counter = 0
    for token in tokens:
        word = clean_str(token)
        if word in emoticon_lexicon_unigram_dict.keys():
            vector_list += emoticon_lexicon_unigram_dict[word]
            counter += 1
    if counter > 0:
        vector_list /= counter
    return poly_emoticon_lexicon.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_bigram_sentiment_emoticon_lexicon_vector(tokens):
    bi_tokens = bigrams(tokens)
    vector_list = np.zeros(3)
    counter = 0
    for bi_token in bi_tokens:
        word = clean_str(" ".join(bi_token))
        if word in emoticon_lexicon_bigram_dict.keys():
            vector_list += emoticon_lexicon_bigram_dict[word]
            counter += 1
    if counter > 0:
        vector_list /= counter
    return poly_emoticon_lexicon.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_pair_sentiment_emoticon_lexicon_vector(tokens):
    vector_list = np.zeros(3)
    counter = 0

    for i in range(len(tokens)):
        word_1 = clean_str(tokens[i])
        if word_1 in emoticon_lexicon_pairs_dict.keys():
            token_1_dict = emoticon_lexicon_pairs_dict[word_1]
            for j in range(i, len(tokens)):
                word_2 = clean_str(tokens[j])
                if word_2 in token_1_dict.keys():
                    vector_list += token_1_dict[word_2]
                    counter += 1

    if counter > 0:
        vector_list /= counter
    return poly_emoticon_lexicon.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_sentiment_emoticon_lexicon_vector(tweet):
    final_list = list()
    tokens = word_tokenize(tweet)

    # Adding unigram features
    final_list.extend(get_unigram_sentiment_emoticon_lexicon_vector(tokens))

    # Adding bigram features
    final_list.extend(get_bigram_sentiment_emoticon_lexicon_vector(tokens))

    # Adding pair features
    final_list.extend(get_pair_sentiment_emoticon_lexicon_vector(tokens))

    return final_list


# ## Emoticon Sentiment Aff-Neg Lexicon

# In[64]:

emoticon_afflex_unigrams_file_path = \
    wassa_home + "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/" + \
    "NRC-Emoticon-Lexicon-v1.0/Emoticon-unigrams.txt"
emoticon_afflex_bigrams_file_path = \
    wassa_home + "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/" + \
    "NRC-Emoticon-Lexicon-v1.0/Emoticon-bigrams.txt"

emoticon_afflex_unigrams = dict()
emoticon_afflex_bigrams = dict()


def get_emoticon_afflex_unigram_dict():
    with open(emoticon_afflex_unigrams_file_path) as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_afflex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])

    return emoticon_afflex_unigrams


def get_emoticon_afflex_bigram_dict():
    with open(emoticon_afflex_bigrams_file_path) as emoticon_lexicon_file:
        for line in emoticon_lexicon_file:
            word_array = line.replace("\n", "").split("\t")
            emoticon_afflex_bigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])

    return emoticon_afflex_bigrams


# In[65]:
print("Loading emoticon_afflex_unigram_dict")
emoticon_afflex_unigram_dict = get_emoticon_afflex_unigram_dict()


# In[66]:
print("Loading emoticon_afflex_bigram_dict")
emoticon_afflex_bigram_dict = get_emoticon_afflex_bigram_dict()


# In[67]:
poly_emoticon_lexicon = PolynomialFeatures(5)


def get_unigram_sentiment_emoticon_afflex_vector(tokens):
    vector_list = np.zeros(3)
    counter = 0
    for token in tokens:
        word = clean_str(token)
        if word in emoticon_afflex_unigram_dict.keys():
            vector_list += emoticon_afflex_unigram_dict[word]
            counter += 1
    if counter > 0:
        vector_list /= counter
    return poly_emoticon_lexicon.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_bigram_sentiment_emoticon_afflex_vector(tokens):
    bi_tokens = bigrams(tokens)
    vector_list = np.zeros(3)
    counter = 0
    for bi_token in bi_tokens:
        word = clean_str(" ".join(bi_token))
        if word in emoticon_afflex_bigram_dict.keys():
            vector_list += emoticon_afflex_bigram_dict[word]
            counter += 1
    if counter > 0:
        vector_list /= counter
    return poly_emoticon_lexicon.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_sentiment_emoticon_afflex_vector(tweet):
    final_list = list()
    tokens = word_tokenize(tweet)

    # Adding unigram features
    final_list.extend(get_unigram_sentiment_emoticon_afflex_vector(tokens))

    # Adding bigram featunigram_list =ures
    final_list.extend(get_bigram_sentiment_emoticon_afflex_vector(tokens))

    return final_list


# ## Hashtag Sentiment Aff-Neg Lexicon

# In[68]:

hashtag_affneglex_unigrams_file_path = \
    wassa_home + "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons" + \
    "/NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-unigrams.txt"
hashtag_affneglex_bigrams_file_path = \
    wassa_home + "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/" + \
    "NRC-Hashtag-Sentiment-AffLexNegLex-v1.0/HS-AFFLEX-NEGLEX-bigrams.txt"

hashtag_affneglex_unigrams = dict()
hashtag_affneglex_bigrams = dict()


def get_hashtag_affneglex_unigram_dict():
    with open(hashtag_affneglex_unigrams_file_path) as hashtag_sent_lex_file:
        for line in hashtag_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            hashtag_affneglex_unigrams[clean_str(word_array[0])] = np.array([float(val) for val in word_array[1:]])

    return hashtag_affneglex_unigrams


def get_hashtag_affneglex_bigram_dict():
    with open(hashtag_affneglex_bigrams_file_path) as hashtag_sent_lex_file:
        for line in hashtag_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            hashtag_affneglex_bigrams[clean_str(word_array[0])] = np.array([float(val) for val in word_array[1:]])

    return hashtag_affneglex_bigrams


# In[69]:
print("Loading hashtag_affneglex_unigram_dict")
hashtag_affneglex_unigram_dict = get_hashtag_affneglex_unigram_dict()


# In[70]:
print("Loading hashtag_affneglex_bigram_dict")
hashtag_affneglex_bigram_dict = get_hashtag_affneglex_bigram_dict()


# In[71]:
poly_hashtag_sent_affneglex = PolynomialFeatures(5)


def get_unigram_sentiment_hashtag_affneglex_vector(tokens):
    vector_list = np.zeros(3)
    counter = 0
    for token in tokens:
        word = clean_str(token)
        if word in hashtag_affneglex_unigram_dict.keys():
            vector_list += hashtag_affneglex_unigram_dict[word]
            counter += 1
    if counter > 0:
        vector_list /= counter
    return poly_hashtag_sent_affneglex.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_bigram_sentiment_hashtag_affneglex_vector(tokens):
    bi_tokens = bigrams(tokens)
    vector_list = np.zeros(3)
    counter = 0
    for bi_token in bi_tokens:
        word = clean_str(" ".join(bi_token))
        if word in hashtag_affneglex_bigram_dict.keys():
            vector_list += hashtag_affneglex_bigram_dict[word]
            counter += 1
    if counter > 0:
        vector_list /= counter
    return poly_hashtag_sent_affneglex.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_sentiment_hashtag_affneglex_vector(tweet):
    final_list = list()
    tokens = word_tokenize(tweet)

    # Adding unigram features
    final_list.extend(get_unigram_sentiment_hashtag_affneglex_vector(tokens))
    # Adding bigram features
    final_list.extend(get_bigram_sentiment_hashtag_affneglex_vector(tokens))

    return final_list


# ## Hashtag Sentiment Lexicon

hash_sent_lex_unigrams_file_path = \
    wassa_home + "lexicons/NRC-Sentiment-Emotion-Lexicons/Lexicons/" + \
    "NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-unigrams.txt"
hash_sent_lex_bigrams_file_path = wassa_home + "lexicons/NRC-Sentiment-Emotion-Lexicons/" + \
    "Lexicons/NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-bigrams.txt"
hash_sent_lex_pairs_file_path = wassa_home + "lexicons/NRC-Sentiment-Emotion-Lexicons/" + \
    "Lexicons/NRC-Hashtag-Sentiment-Lexicon-v1.0/HS-pairs.txt"
pair_split_string = "---"

hash_sent_lex_unigrams = dict()
hash_sent_lex_bigrams = dict()
hash_sent_lex_pairs = dict()


def get_hash_sent_lex_unigram_dict():
    with open(hash_sent_lex_unigrams_file_path) as hash_sent_lex_file:
        for line in hash_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            if clean_str(word_array[0]):
                hash_sent_lex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])

    return hash_sent_lex_unigrams


def get_hash_sent_lex_bigram_dict():
    with open(hash_sent_lex_bigrams_file_path) as hash_sent_lex_file:
        for line in hash_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            if clean_str(word_array[0]):
                hash_sent_lex_unigrams[word_array[0]] = np.array([float(val) for val in word_array[1:]])

    return hash_sent_lex_bigrams


def get_hash_sent_lex_pairs_dict():
    with open(hash_sent_lex_pairs_file_path) as hash_sent_lex_file:
        for line in hash_sent_lex_file:
            word_array = line.replace("\n", "").split("\t")
            pair = word_array[0].split(pair_split_string)
            token_1 = clean_str(pair[0])
            token_2 = clean_str(pair[1])
            if token_1 and token_2:
                token_1_dict = None
                if token_1 in hash_sent_lex_pairs.keys():
                    token_1_dict = hash_sent_lex_pairs[token_1]
                else:
                    token_1_dict = dict()

                token_1_dict[token_2] = np.array([float(val) for val in word_array[1:]])
                hash_sent_lex_pairs[token_1] = token_1_dict

    return hash_sent_lex_pairs


# In[76]:

print("Loading hash_sent_lex_unigram_dict")
hash_sent_lex_unigram_dict = get_hash_sent_lex_unigram_dict()


# In[77]:

print("Loading hash_sent_lex_bigram_dict")
hash_sent_lex_bigram_dict = get_hash_sent_lex_bigram_dict()


# In[78]:

print("Loading hash_sent_lex_pairs_dict")
hash_sent_lex_pairs_dict = get_hash_sent_lex_pairs_dict()


# In[89]:

poly_hash_sent_lex = PolynomialFeatures(5)


def get_unigram_sentiment_hash_sent_lex_vector(tokens):
    vector_list = np.zeros(3)
    counter = 0

    for token in tokens:
        word = clean_str(token)
        if word in hash_sent_lex_unigram_dict.keys():
            vector_list += hash_sent_lex_unigram_dict[word]
            counter += 1

    if counter > 0:
        vector_list /= counter

    return poly_hash_sent_lex.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_bigram_sentiment_hash_sent_lex_vector(tokens):
    bi_tokens = bigrams(tokens)
    vector_list = np.zeros(3)
    counter = 0
    for bi_token in bi_tokens:
        word = clean_str(" ".join(bi_token))
        if word in hash_sent_lex_bigram_dict.keys():
            vector_list += hash_sent_lex_bigram_dict[word]
            counter += 1
    if counter > 0:
        vector_list /= counter

    return poly_hash_sent_lex.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_pair_sentiment_hash_sent_lex_vector(tokens):
    vector_list = np.zeros(3)
    counter = 0

    for i in range(len(tokens)):
        word_1 = clean_str(tokens[i])
        if word_1 in hash_sent_lex_pairs_dict.keys():
            token_1_dict = hash_sent_lex_pairs_dict[word_1]
            for j in range(i, len(tokens)):
                word_2 = clean_str(tokens[j])
                if word_2 in token_1_dict.keys():
                    vector_list += token_1_dict[word_2]
                    counter += 1
    if counter > 0:
        vector_list /= counter
    return poly_hash_sent_lex.fit_transform(vector_list.reshape(1, -1))[0].tolist()


def get_sentiment_hash_sent_lex_vector(tweet):
    final_list = list()

    tokens = word_tokenize(tweet)

    # Adding unigram features
    final_list.extend(get_unigram_sentiment_hash_sent_lex_vector(tokens))
    # Adding bigram features
    final_list.extend(get_bigram_sentiment_hash_sent_lex_vector(tokens))
    # Adding pair features
    final_list.extend(get_pair_sentiment_hash_sent_lex_vector(tokens))

    return final_list


# # Automated Testing

# In[225]:

num_test = 32767


# In[226]:

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

    vectors = pd.concat(frames, axis=1)

    return vectors.values.tolist()


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

    score_train = list()
    tweet_train = list()
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


for emotion in ['anger', 'sadness', 'joy', 'fear']:

    print("Working on: " + emotion)
    tweet_train, tweet_test, score_train, y_gold = load_all_data(emotion)

    print("Loading word_intensities")
    word_intensities = get_word_affect_intensity_dict(emotion)
    print("Loading affect_presence_word_list")
    affect_presence_word_list = get_affect_presence_list(emotion)
    print("Loading hashtag_emotion_intensities")
    hashtag_emotion_intensities = get_hashtag_emotion_intensity(emotion)

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

    train_vector_dict = dict()
    test_vector_dict = dict()

    for i in range(num_test, 0, -1):
        print("Current test: " + str(i) + "/" + str(num_test))
        bin_string = '{0:015b}'.format(i)
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
