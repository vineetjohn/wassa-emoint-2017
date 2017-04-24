from gensim.models.doc2vec import TaggedDocument


class LabeledLineSentence(object):

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            yield TaggedDocument(line.split(), ['SENT_%s' % uid])
