LSTM for Twitter Sentiment Analysis
=============================================


## Files

1. `sentiment140_clean.py` -- script to clean original tweets.
2. `transform.py` -- script to transform words into numbers with count-based approach.
3. `tweet.py` -- script to define methods to prepare and load data.
4. `tweet_lstm.py` -- script to train LSTM on the tweets.


## Execution
First, please make sure that you have installed all of the dependencies listed in
`requirements.txt` file. Then, type

```
  python sentiment140_clean.py
  python transform.py
  THEANO_FLAGS="floatX=float32" python tweet_lstm.py
```

to execute the scripts. It may cost about 40 minutes to complete the training, depending on
the concrete hardware configuration.
