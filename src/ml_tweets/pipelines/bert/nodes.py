"""
This is a boilerplate pipeline 'bert'
generated using Kedro 0.18.1
"""
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict,Any,Tuple
from .data_wrapper import create_data_loader_train, create_data_loader_test
import logging
from .model import BertClassifier
from .trainer import train_epoch, eval_model

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


if torch.cuda.is_available():
    device= torch.device('cuda')
    logging.info(f"There are {torch.cuda.device_count()} available GPUs" )
    logging.info(f" Device name: {torch.cuda.get_device_name(0)}")
else:
    device= torch.device("cpu")



PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
#load tokenizer
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BATCH_SIZE = 16

CUSTOM_MAX_LEN=60

# find max length of tweets
def find_max_tweet_length(df:pd.DataFrame) -> int:
    token_lens = []
    for txt in df.text:
        tokens = tokenizer.encode(txt, max_length=512)
        token_lens.append(len(tokens))
        max_len = max(token_lens)
    return max_len


def prepare_data(
    train_df: pd.DataFrame
) -> Tuple[DataLoader, DataLoader]:
    
    #split data
    train_df, val_df = train_test_split(train_df, test_size=0.1,random_state=RANDOM_SEED)
    logging.info(f"Train Shape: {train_df.shape}, Validation Shape: {val_df.shape}")

    #max_len= find_max_tweet_length(train_df)

    #define data loaders
    train_dataloader = create_data_loader_train(train_df, tokenizer, CUSTOM_MAX_LEN, BATCH_SIZE)
    val_dataloader = create_data_loader_train(val_df, tokenizer, CUSTOM_MAX_LEN, BATCH_SIZE)
    return val_dataloader, val_dataloader,



def fit(train_dataloader: DataLoader, val_dataloader: DataLoader)-> any:
    #define bert model
    n_classes = 2
    dropout_p = 0.2
    model= BertClassifier(PRE_TRAINED_MODEL_NAME, n_classes, dropout_p)
    model.to(device)
    epochs = 10
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0
    
    logging.info(f"Training Entries: {len(train_dataloader.dataset)}")
    
    
    logging.info(f"----Start Training ----")
    #--- Train the model ---#
    for epoch in range(epochs):
        logging.info(f'Epoch {epoch + 1}/{epochs}')
        logging.info('-' * 10)
        
        train_acc, train_loss = train_epoch(
                                model,
                                train_dataloader,
                                loss_fn,
                                optimizer,
                                device,
                                scheduler,
                                len(train_dataloader.dataset)
                            )
        
        logging.info(f'Train: loss {train_loss} -  accuracy {train_acc}')
        
        val_acc, val_loss = eval_model(
                                model,
                                val_dataloader,
                                loss_fn,
                                device,
                                len(val_dataloader.dataset)
                            )
        
        logging.info(f'Validation: loss {val_loss} - accuracy {val_acc}')
        
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    return model


def submit(model:any,  test_df: pd.DataFrame, parameters: Dict[str, Any]):
    sample_submission = pd.read_csv(parameters['sample_submission'])
    
    data_loader= create_data_loader_test(test_df, tokenizer, CUSTOM_MAX_LEN, BATCH_SIZE)
    
    model = model.eval()
    tweets = []
    predictions = []
    prediction_probs = []
    
    with torch.no_grad():
        for data in data_loader:
            texts = data["text"]
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
        
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
            _, preds = torch.max(outputs, dim=1)
            tweets.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
        
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    
    assert len(sample_submission) == len(predictions)
    submission = pd.DataFrame({'id':sample_submission['id'].values.tolist(), "target": predictions.tolist()})
    return submission
