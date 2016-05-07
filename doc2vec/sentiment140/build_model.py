# -*- coding: utf-8 -*-
"""
    build_model.py
    ~~~~~~~~~~~~~~~~~

    This file defines methods to build doc2vec models from files with positive and negative
    tweets for training and testing. The data is from sentiment140 data set.

    The script is based on the approach show in
    https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb
"""

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import numpy
import random
import logging
import os.path
import sys
import cPickle as pickle


program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

# logger.setLevel(logging.INFO)
#
# handler = logging.FileHandler('build_model.log')
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

logger.info("running %s" % ' '.join(sys.argv))


# This is a class which is necessary for doc2vec() method.
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        flipped = {}
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as f:
                for item_no, line in enumerate(f):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        random.shuffle(self.sentences)
        return self.sentences


# Please notice that here, I also include test files in training language model.
sources = {'train_pos.txt': 'TRAIN_POS', 'train_neg.txt': 'TRAIN_NEG',
           'test_pos.txt': 'TEST_POS', 'test_neg.txt': 'TEST_NEG'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=10, size=100, dm_mean=0, sample=1e-5, negative=5, workers=12)

model.build_vocab(sentences.to_array())

for epoch in range(20):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())

model.save('./sentiment140.d2v')
