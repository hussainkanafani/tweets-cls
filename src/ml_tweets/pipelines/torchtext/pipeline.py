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
                outputs=["train_loader","valid_loader","vocab_size","text_pipeline"],
                name="prepare_data",
            ),
            node(
                func=fit_and_evaluate,
                inputs=["train_loader","valid_loader","vocab_size","text_pipeline","sample_submission","kaggle_submission"],
                outputs="torch_model",
                name="fit_and_evaluate",
            ),
            #node(
            #    func=evaluate,
            #    inputs=["model","sample_submissions"],
            #    outputs=[""],
            #    name="fit",
            #)
    ])
