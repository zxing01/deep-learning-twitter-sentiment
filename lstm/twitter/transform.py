# -*- coding: utf-8 -*-
"""
    transform.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    This file defines the methods transfrom words into numbers represented by the
    count of occurence in the text corpus.
"""


import numpy
from six.moves import cPickle


def build_dict(file_names):
    contents = []
    for file_name in file_names:
        with open(file_name, 'r') as f:
            for line in f:
                contents.append(line)

    word_dict = dict()
    for line in contents:
        words = line.split()
        for w in words:
            if w not in word_dict:
                word_dict[w] = 1
            else:
                word_dict[w] += 1

    counts = word_dict.values()
    keys = word_dict.keys()

    sorted_idx = numpy.argsort(counts)[:-1]
    # print "sorted_idx", sorted_idx

    worddict = dict()
    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx + 2

    print numpy.sum(counts), 'total words', len(keys), 'unique words'
    return worddict


def grab_data(path, dictionary):
    sents = []
    with open(path, 'r') as f:
        sents = f.readlines()

    seqs = [None] * len(sents)
    for idx, ss in enumerate(sents):
        words = ss.split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    # print seqs
    return seqs


if __name__ == '__main__':
    word_dict = build_dict(['train_pos.txt', 'train_neg.txt'])

    train_x_pos = grab_data('train_pos.txt', word_dict)
    train_x_neg = grab_data('train_neg.txt', word_dict)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data('test_pos.txt', word_dict)
    test_x_neg = grab_data('test_neg.txt', word_dict)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('tweet.pkl', 'wb')
    cPickle.dump((train_x, train_y), f, -1)
    cPickle.dump((test_x, test_y), f, -1)
    f.close()

    f = open('tweet.dict.pkl', 'wb')
    cPickle.dump(word_dict, f, -1)
    f.close()
