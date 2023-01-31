"""
This is a boilerplate pipeline 'bow_linear'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess, fit_and_evaluate

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess,
                inputs=["tweets_train","tweets_test"],
                outputs=["train_vectors","test_vectors","targets"],
                name="preprocess",
            ),
            node(
                func=fit_and_evaluate,
                inputs=["train_vectors","test_vectors","targets","parameters"],
                outputs="y_pred",
                name="fit_and_evaluate",
            )
        ]
    )
