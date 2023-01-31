import torch
from torch import nn
import numpy as np


def train_epoch(
      model,
      data_loader,
      loss_fn,
      optimizer,
      device,
      scheduler,
      n_examples
):
      model = model.train()
      losses = []
      correct_predictions = 0
        
      for data in data_loader:
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            targets = data["targets"].to(device)

            # inject inputs ids and attention mask into bert model 
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask
            )

            # return prediction 
            _, preds = torch.max(outputs, dim=1)
            #print(outputs.size())
            #print(targets.size())
            #print(outputs.type())
            #print(targets.type())
            
            # calculate loss 
            loss = loss_fn(outputs, targets)

            # calculate correct prediction 
            correct_predictions += torch.sum(preds == targets)
            
            # append each loss
            losses.append(loss.item())
            
            # back propagation 
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # update parameters 
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
      return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for data in data_loader:
      input_ids = data["input_ids"].to(device)
      attention_mask = data["attention_mask"].to(device)
      targets = data["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
        
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
        
  return correct_predictions.double() / n_examples, np.mean(losses)


  