from torch import nn
import torch
from transformers import BertModel

class BertClassifier(nn.Module):
  def __init__(self, pretrained_model_name, n_classes, p):
    super(BertClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(pretrained_model_name)
    self.hidden = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
    self.drop = nn.Dropout(p)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.softmax = nn.Softmax(dim=1)
    #init weights
    torch.nn.init.xavier_uniform_(self.hidden.weight)
    torch.nn.init.xavier_uniform_(self.out.weight)

    
  def forward(self, input_ids, attention_mask):
    output_dict = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    hidden=self.hidden(output_dict['pooler_output'])
    output = self.drop(hidden)
    output = self.out(output)
    return self.softmax(output)