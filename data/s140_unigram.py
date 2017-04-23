class S140Unigram(object):

    def __init__(self, word, score):
        self.word = word
        self.score = score

    def __repr__(self):
        return \
            "word: " + self.word + \
            ", score: " + self.score
