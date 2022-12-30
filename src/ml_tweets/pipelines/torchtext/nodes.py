"""
This is a boilerplate pipeline 'torchtext'
generated using Kedro 0.18.1
"""
import logging
import re as re
from typing import Any, Dict, List, Tuple

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from .data_wrapper import (
    DataWrapper,
    collate_batch,
    tokenizer
)
from .model import TextClassificationModel
from .trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_data(
    train_df: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, int]:

    BATCH_SIZE = 64 # batch size for training
    
    logging.info("collate dataloaders on train_df")
    train_dataset = DataWrapper(train_df)
    logging.info(train_dataset.df.head())

    text_pipeline= train_dataset.text_pipeline
    label_pipeline= train_dataset.label_pipeline
    
    
    num_train = int(len(train_dataset) * 0.90)
    
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=BATCH_SIZE, collate_fn=lambda batch: collate_batch(batch, text_pipeline, label_pipeline))
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=BATCH_SIZE, collate_fn=lambda batch: collate_batch(batch, text_pipeline, label_pipeline))

    return train_dataloader,valid_dataloader, train_dataset.vocab


def fit(train_dataloader: DataLoader, valid_dataloader: DataLoader, vocab)-> any:
    
    EPOCHS = 30 # epoch
    LR = 1  # learning rate
    
    #model params
    num_class = 1  
    emsize=64
    
    vocab_size= len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    trainer = Trainer (
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        logger_kwargs={'update_step':20, 'show':True},
        device=device
    )

    trainer.fit(train_dataloader, valid_dataloader, EPOCHS)
    
    return trainer.model

def submit(model:any, vocab:any, test_df: pd.DataFrame, parameters: Dict[str, Any]):
    sample_submission = pd.read_csv(parameters['sample_submission'])

    predictions=[]
    with torch.no_grad():
        for text in test_df.text:
            text = torch.tensor(vocab(tokenizer(text)))
            output = model(text, torch.tensor([0]))
            predictions.append(round(output.item()))
    #print(predictions)
    assert len(sample_submission) == len(predictions)
    #pd.DataFrame({'id':sample_submission['id'].values.tolist(), "target": predictions}).to_csv('submission.csv', index=False, header=True)
    submission = pd.DataFrame({'id':sample_submission['id'].values.tolist(), "target": predictions})
    return submission
