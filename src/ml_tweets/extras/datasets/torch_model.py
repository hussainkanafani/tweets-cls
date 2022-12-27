from typing import Any, Dict

import numpy as np

from kedro.io import AbstractDataSet, AbstractVersionedDataSet
from ml_tweets.pipelines.torchtext.model import TextClassificationModel
import torch
from pathlib import PurePosixPath,Path

MODELS={'torchtext_classification': "TextClassificationModel"}

class TorchModelDataSet(AbstractVersionedDataSet):
    """
    ``TorchModelDataSet`` loads / save torch model.
    """

    def __init__(self, filepath: str, version, model, load_args= None, save_args=None):
        """Creates a new instance of ImageDataSet to load / save image data at the given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        super().__init__(PurePosixPath(filepath),version)
        #self._filepath = filepath
        model =model.lower()

        if model in MODELS:
            self._model_name= model
            self._model=MODELS[model]
        else:
            raise KeyError(f"unknown model: {model}")

        default_load_args={}
        default_save_args={}

        self._load_args=({**default_load_args,**load_args} if load_args is not None else {**default_load_args})
        self._save_args=({**default_save_args,**save_args} if save_args is not None else {**default_save_args})

    def _load(self) -> np.ndarray:
        """Loads model from path of state dict.

        Returns:
            Model from given path.
        """
        filepath=PurePosixPath(self._get_load_path())
        state_dict=torch.load(filepath)
        model= self._model(**self._load_args)
        model.load_state_dict(state_dict)
        return model


    def _save(self, model: any) -> None:
        """Saves model data to the specified filepath"""
        filepath=Path(self._get_save_path())
        print(f"filepath: {filepath}")
        print(f"filepath: {list(filepath.parents)}")
        print(model)        

        if not filepath.parents[0].exists():
            filepath.parents[0].mkdir(parents=True)
    
        torch.save(model.state_dict(),str(filepath),**self._save_args)


    def _describe(self) -> Dict[str, Any]:
        return dict(version=self._version)

    def _exists(self) -> bool:
        path= self._get_load_path()
        return Path(path.as_posix()).exists()