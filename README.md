# Disaster Tweets Classifier

## Overview

This project solves the [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) on [Kaggle](https://www.kaggle.com/) In this repository different types of machine learning algorithms are tried out to predict which Tweets are about real disasters and which ones are not.
Data pipelines for the following models:

  * [RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html) with bag of words.
  * Custom [Torchtext](https://pytorch.org/text/stable/index.html) model
  * Fine tuned [Huggingface Bert](https://huggingface.co/docs/transformers/model_doc/bert). (ACHIEVES BEST RESULT ON KAGGLE WITH SCORE: `0.83358`
  * TODO: LSTM


This is your new Kedro project, which was generated using [Kedro 0.18.1](https://kedro.readthedocs.io).

## Instructions

To install requirements, run:

```
pip install -r src/requirements.txt
```

You can run any of the defined kedro pipelines with:

```
kedro run --pipeline pipeline-name 

e.g.: kedro run --pipeline bert 

```

