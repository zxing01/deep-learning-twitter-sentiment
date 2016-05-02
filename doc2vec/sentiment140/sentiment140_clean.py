# -*- coding: utf-8 -*-
"""
    sentiment140_clean.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    This file defines the methods to clean sentiment140 training and testing data into
    sentences with clean tokens.
"""


from string import punctuation
import re
from nltk.tokenize import TweetTokenizer
import random


def write_to_file(file_name, sents):
    with open(file_name, 'w') as f:
        for sent in sents:
            #print sent
            #f.write(sent.encode('utf-8'))
            try:
                f.write(sent)
            except UnicodeEncodeError:
                continue
            f.write('\n')


def clean_tweet(tweet):
    tknzr = TweetTokenizer()
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet.lower())
    tweet = ' '.join(tweet.split())
    words = tknzr.tokenize(tweet)
    words = [''.join(c for c in s if c not in punctuation) for s in words]
    words = [s for s in words if s]
    sent = " ".join(words)
    return sent


def clean_data(input_file_name, output_file_name):
    with open(input_file_name, 'r') as f:
        content = f.readlines()

    # Please notice that only part of the tweets are used.
    if len(content) > 1000:
        random.shuffle(content)
        content = content[0:100000]  # To improve accuracy, I need to adjust the number.
    tweet_sents = []
    for line in content:
        try:
            sent = clean_tweet(line)
            tweet_sents.append(sent)
        except UnicodeDecodeError:
            continue

    write_to_file(output_file_name, tweet_sents)


if __name__ == '__main__':
    clean_data('../../sentiment140/train.pos.txt', 'train_pos.txt')
    clean_data('../../sentiment140/train.neg.txt', 'train_neg.txt')
    clean_data('../../sentiment140/test.pos.txt', 'test_pos.txt')
    clean_data('../../sentiment140/test.neg.txt', 'test_neg.txt')
