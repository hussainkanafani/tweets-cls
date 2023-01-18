from torch.utils.data import Dataset, DataLoader
import torch



#---- Dataset for Train Data ----#
class TweetDataset(Dataset):
  def __init__(self, text, targets, tokenizer, max_len):
    self.text = text
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    
  def __len__(self):
    return len(self.text)


  def __getitem__(self, item):
    text = str(self.text[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    
    
    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


#---- Dataset for Test Data ----#
#TODO make one dataset for both train and test 

class TweetDatasetTest(Dataset):
  def __init__(self, text, tokenizer, max_len):
    self.text = text
    self.tokenizer = tokenizer
    self.max_len = max_len
    
  def __len__(self):
    return len(self.text)


  def __getitem__(self, item):
    text = str(self.text[item])
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt'
    )
    
    
    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten()
    }


#--- Data Loader for Train Data ---#

def create_data_loader_train(df, tokenizer, max_len, batch_size):
  dataset = TweetDataset(text=df.text.to_numpy(),targets=df.target.to_numpy(),tokenizer=tokenizer, max_len=max_len)

  return DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=2
  )

#--- Data Loader for Test Data ---#

def create_data_loader_test(df, tokenizer, max_len, batch_size):
  dataset = TweetDatasetTest(
    text=df.text.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=2
  )
