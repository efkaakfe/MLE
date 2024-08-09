"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from typing import List

import pandas as pd

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

def inference(model, batch, test):
  test_loader = DataLoader(test, batch_size = batch, shuffle = True)
  test_loss = 0
  test_acc = 0
  loss_fn = nn.CrossEntropyLoss()
  model.eval()
  with torch.no_grad():
      for batch in test_loader:
          x = batch[:,:-1]
          y = batch[:,-1].reshape(-1,1)
          out = model(x)
          loss = loss_fn(out, y)
          test_loss += loss.item()
          pred = torch.argmax(out, 1).detach().numpy()
          test_acc += accuracy_score(y, pred)


  test_loss /= len(test_loader)
  test_acc /= len(test_loader)
  return test_loss, test_acc

def main():
    X = pd.read_csv('./data_process/test.csv')
    y = pd.read_csv('./data_process/ytest.csv')
    model = joblib.load('./model/model.joblib')
    

if __name__ == "__main__":
    main()