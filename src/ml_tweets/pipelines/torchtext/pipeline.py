"""
This is a boilerplate pipeline 'torchtext'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_data, fit, submit



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
            func=prepare_data,
            inputs=["tweets_train", "parameters"],
            outputs=["train_loader","valid_loader","vocab_pkl"],
            name="prepare_data",
            ),
            node(
                func=fit,
                inputs=["train_loader","valid_loader","vocab_pkl"],
                outputs="torch_model",
                name="fit",
            ),
            node(
                func=submit,
                inputs=["torch_model","vocab_pkl","tweets_test","parameters"],
                outputs="kaggle_submission",
                name="submit",
            )
    ])
