# -*- coding: utf-8 -*-
"""
    process.py
    ~~~~~~~~~~~~~

    This file defines methods to implement sentiment analysis on Doc2Vec model which is
    trained through build_model.py file.
    This file contains the following classifiers:
        Logistic Regression
        Support Vector Machine
        Random Forest
        K-Nearest Neighbors

    The content of this file is based on the reference:
    https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb

    The usage is this script is:
        python process.py [train_pos_count] [train_neg_count] -[lr/svm/rf/knn]
"""


import sys
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


model = Doc2Vec.load('./sentiment140.d2v')

if len(sys.argv) < 4:
    print "Please input train_pos_count, train_neg_count and classifier!"
    sys.exit()

train_pos_count = int(sys.argv[1])
train_neg_count = int(sys.argv[2])
test_pos_count = 182
test_neg_count = 177

# print train_pos_count
# print train_neg_count

vec_dim = 100

print "Build training data set..."
train_arrays = numpy.zeros((train_pos_count + train_neg_count, vec_dim))
train_labels = numpy.zeros(train_pos_count + train_neg_count)

for i in range(train_pos_count):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_labels[i] = 1

for i in range(train_neg_count):
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[train_pos_count + i] = model.docvecs[prefix_train_neg]
    train_labels[train_pos_count + i] = 0


print "Build testing data set..."
test_arrays = numpy.zeros((test_pos_count + test_neg_count, vec_dim))
test_labels = numpy.zeros(test_pos_count + test_neg_count)

for i in range(test_pos_count):
    prefix_test_pos = 'TEST_POS_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_labels[i] = 1

for i in range(test_neg_count):
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[test_pos_count + i] = model.docvecs[prefix_test_neg]
    test_labels[test_pos_count + i] = 0


print "Begin classification..."
classifier = None
if sys.argv[3] == '-lr':
    print "Logistic Regressions is used..."
    classifier = LogisticRegression()
elif sys.argv[3] == '-svm':
    print "Support Vector Machine is used..."
    classifier = SVC()
elif sys.argv[3] == '-knn':
    print "K-Nearest Neighbors is used..."
    classifier = KNeighborsClassifier(n_neighbors=10)
elif sys.argv[3] == '-rf':
    print "Random Forest is used..."
    classifier = RandomForestClassifier()

classifier.fit(train_arrays, train_labels)

print "Accuracy:", classifier.score(test_arrays, test_labels)
