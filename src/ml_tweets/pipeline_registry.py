"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from ml_tweets.pipelines import tfidf_lstm, bow_linear

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    return {
        "__default__": bow_linear.create_pipeline(), # set bag of words wit linear to default
        "bow_linear": bow_linear.create_pipeline(), 
        "tfidf_lstm": tfidf_lstm.create_pipeline(),
    }
