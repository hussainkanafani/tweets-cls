"""
This is a boilerplate pipeline 'bow_linear'
generated using Kedro 0.18.1
"""
"""
This is a boilerplate pipeline
generated using Kedro 0.18.1
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing


def preprocess(
    train_df: pd.DataFrame, test_df: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    logging.info("init CountVectorizer")

    count_vectorizer = feature_extraction.text.CountVectorizer()

    logging.info("fit CountVectorizer on train_df")
    train_vectors = count_vectorizer.fit_transform(train_df["text"])
    
    logging.info("fit CountVectorizer on test_df")
    test_vectors = count_vectorizer.transform(test_df["text"])
    
    return train_vectors, test_vectors,train_df['target']


def fit_and_evaluate(
    train_vectors: pd.DataFrame,test_vectors: pd.DataFrame, targets: pd.Series, parameters: Dict[str, Any]) -> pd.Series:
    
    clf = linear_model.RidgeClassifier()
    logging.info("create classifier")
    scores = model_selection.cross_val_score(clf, train_vectors, targets, cv=parameters['cv'], scoring=parameters['scoring'])
    
    logging.info("fit classifier on train_vectors")
    clf.fit(train_vectors, targets)
    
    sample_submission = pd.read_csv(parameters['sample_submission'])
    
    logging.info("predict targets of test data")
    sample_submission["target"] = clf.predict(test_vectors)

    logging.info(f"sample submissions look like this: {sample_submission.head()}")

    logging.info(f"save submission dataframe to {parameters['kaggle_submissions']}")
    sample_submission.to_csv(parameters['kaggle_submissions'], index=False)
    
    return sample_submission


