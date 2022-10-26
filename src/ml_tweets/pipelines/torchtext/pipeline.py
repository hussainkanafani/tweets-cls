"""
This is a boilerplate pipeline 'torchtext'
generated using Kedro 0.18.1
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import prepare_data, fit_and_evaluate



def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
                node(
                func=prepare_data,
                inputs=["tweets_train", "parameters"],
                outputs=["train_loader","valid_loader","vocab_size"],
                name="prepare_data",
            ),
            node(
                func=fit_and_evaluate,
                inputs=["train_loader","valid_loader","vocab_size"],
                outputs=["train_loss","val_loss"],
                name="fit_and_evaluate",
            )
    ])
