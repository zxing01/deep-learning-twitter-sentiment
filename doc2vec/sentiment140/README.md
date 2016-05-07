Doc2vec for Twitter Sentiment Analysis
===============================================


## Files

1. `build_model.py` -- script to train paragraph vector model.
2. `process.py` -- script to classify positive/negative tweets.
3. `sentiment140_clean.py` -- script to clean original tweets.
4. `run.sh` -- script to execute whole process.


## Execution

First, please make sure that you have installed all of the dependencies illustrated in
`requirements.txt` file. Then, type

```
  bash run.sh
```

to execute the script. During the training, the details will be output onto the screen.
After the execution, you can run `process.py` file for the classification.
The necessary parameters for that script is defined in `process.py` file. You need to provide
information like

```
  python process.py [train_pos_count] [train_neg_count] -[lr/svm/rf/knn]
```

Please notice that the whole process may be very **TIME-CONSUMING** (1-2 hours)!
