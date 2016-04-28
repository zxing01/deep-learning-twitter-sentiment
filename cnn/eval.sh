#!/bin/bash

python eval.py \
--pos_file=../sentiment140/test.pos.txt \
--neg_file=../sentiment140/test.neg.txt \
--vocab_file=./word2vec/vocab.txt \
--checkpoint_dir=./checkpoints
