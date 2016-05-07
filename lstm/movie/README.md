LSTM for Movie Review Sentiment Analysis
==================================================


## Files

1. `imdb.py` -- script to define methods to prepare and load data.
2. `lstm.py` -- script to train LSTM on the movie reviews.


## Execution
First, please make sure that you have installed all of the dependencies listed in
`requirements.txt` file. Then, type

```
  THEANO_FLAGS="floatX=float32" python lstm.py
```

to execute the script. It may cost about 1 hour to complete the training, depending on
the concrete hardware configuration.
