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
    get_vocab,
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

    text_pipeline,label_pipeline,vocab_size=get_vocab(train_dataset)
    
    num_train = int(len(train_dataset) * 0.95)
    
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    #train_dataloader = collate_data_loader(split_train_,BATCH_SIZE,shuffle=True)
    #valid_dataloader = collate_data_loader(split_valid_,BATCH_SIZE,shuffle=True)
    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=BATCH_SIZE, collate_fn=lambda batch: collate_batch(batch, text_pipeline, label_pipeline))
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=BATCH_SIZE, collate_fn=lambda batch: collate_batch(batch, text_pipeline, label_pipeline))

    return train_dataloader,valid_dataloader,vocab_size,text_pipeline


def fit_and_evaluate(train_dataloader: DataLoader, valid_dataloader: DataLoader, vocab_size: int, text_pipeline:any, sample_submission:pd.DataFrame,kaggle_submission: pd.DataFrame)-> Tuple[(List,List)]:
    
    EPOCHS = 1 # epoch
    LR = 5  # learning rate
    
    #model params
    num_class = 1  
    emsize=64
    

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

    train_loss,val_loss=trainer.fit(train_dataloader, valid_dataloader, EPOCHS)
    
    #return train_loss,val_loss
    return trainer.model
    
    #logging.info("predict target of test data")
    #texts= ["Typhoon Soudelor kills 28 in China and Taiwan",
    #"We're shaking...It's an earthquake", 
    #"They'd probably still show more life than Arsenal did yesterday, eh? EH?",
    #"Hey! How are you",
    #"What a nice hat",
    #"Fuck off!"]
    #with torch.no_grad():
    #    for text in texts:
    #        text = torch.tensor(text_pipeline(text))
    #        output = trainer.model(text, torch.tensor([0]))
    #        logging.info(f"MODEL OUTPUT: {output}")
        

    #logging.info("predict targets of test data")
    #sample_submission["target"] = clf.predict(test_vectors)
#
    #logging.info(f"sample submissions look like this: {sample_submission.head()}")
#
    #logging.info(f"save submission dataframe to {parameters['kaggle_submissions']}")
    #sample_submission.to_csv(parameters['kaggle_submissions'], index=False)
    
