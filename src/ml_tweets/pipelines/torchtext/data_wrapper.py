from torch.utils.data import Dataset, DataLoader

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Callable
import torch
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer('basic_english')

class DataWrapper(Dataset):
 
  def __init__(self,df:pd.DataFrame):
    self.df = df

  def __len__(self):
    return len(self.df)
   
  def __getitem__(self,idx):
    return self.df.iloc[idx]['text'], self.df.iloc[idx]['target']


def yield_tokens(data_iter):
    for idx, (text, label) in enumerate(data_iter):
      yield tokenizer(text)
     

def collate_batch(batch,text_pipeline:Callable, label_pipeline:Callable):
    label_list, text_list, offsets = [], [], [0]
    for (_text,_label) in batch:
         label_list.append(int(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)  




def get_vocab(dataset):
      vocab = build_vocab_from_iterator(yield_tokens(iter(dataset)), specials=["<unk>"])
      vocab.set_default_index(vocab["<unk>"])
      print(len(vocab))
      text_pipeline = lambda x: vocab(tokenizer(x))
      label_pipeline = lambda x: int(x) - 1
      return text_pipeline,label_pipeline,len(vocab)
