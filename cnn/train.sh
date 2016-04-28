#!/bin/bash

python train.py \
--pos_file=../sentiment140/train.pos.txt \
--neg_file=../sentiment140/train.neg.txt \
--word2vec_file=./word2vec/model.ckpt-2265405 \
--vocab_file=./word2vec/vocab.txt \
--checkpoint_dir=./checkpoints
