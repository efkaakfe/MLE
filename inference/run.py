"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""
import os
import time 
import numpy as np
import pandas as pd
import logging 
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Setting the seed for reproductibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Logging 
logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

class IrisNetwork(nn.Module):
  def __init__(self, n_layers, sub):
    super().__init__()
    self.n_layers = n_layers
    self.sub = sub
    self.layers = []
    self.activation = []
    start = 4
    out = 0
    for i in range(1, self.n_layers+1):
      out = start - self.sub
      if i < n_layers:
        self.layers.append(nn.Linear(start, out))
        self.activation.append(nn.ReLU())
        self.add_module(f"layer{i}", self.layers[-1])
        self.add_module(f"act{i}", self.activation[-1])
        start = out
      else:
        self.layers.append(nn.Linear(start, 3))
        self.activation.append(nn.Softmax(dim = 1))
        self.add_module(f"layer{i}", self.layers[-1])
        self.add_module(f"act{i}", self.activation[-1])

  def forward(self, x):
    for i in range(0, self.n_layers):
      x = self.layers[i](x)
      x = self.activation[i](x)
    return x

class TensorTransformer(BaseEstimator, TransformerMixin):
  def fit(self, X):
    return self
  def transform(self, X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor

prep = Pipeline([
    ('scaler', StandardScaler()),
    ('tensor', TensorTransformer())
])

def load_file():
   try:
    with open('./model/batch_size.txt', 'r') as f:
       b = f.read()
       f.close
    return b
   except FileNotFoundError:
     print('No such file')

def save_results(true, pred):
   os.mkdir('./results')
   r = pd.DataFrame({
     'Actual': true,
     'Predicted': pred
   })
   r.to_csv('./results/results.csv', index = False)

def inference(model, batch, test):
  test_loader = DataLoader(test, batch_size = batch)
  true = test[:,-1].detach().numpy()
  test_loss = 0
  test_acc = 0
  loss_fn = nn.CrossEntropyLoss()
  prediction = np.array([])
  model.eval()
  s = time.time()
  logging.info(f'Running inference with {batch} batch size and {test.shape[0]} test size.')
  with torch.no_grad():
      for batch in test_loader:
          x = batch[:,:-1]
          y = batch[:,-1]
          y = y.type(torch.LongTensor)
          out = model(x)
          loss = loss_fn(out, y)
          test_loss += loss.item()
          pred = torch.argmax(out, 1).detach().numpy()
          test_acc += accuracy_score(y, pred)
          prediction = np.concatenate((prediction,pred))

  e = time.time()

  test_loss /= len(test_loader)
  test_acc /= len(test_loader)

  logging.info(f'Inference completed in {e-s} seconds.\nAccuracy score: {test_acc}.')
  save_results(true, prediction)

def main():

    try: 
      X = pd.read_csv('./data_process/test.csv')
    except FileNotFoundError:
       print('No data file')
    
    X = prep.fit_transform(X)

    try: 
      y = pd.read_csv('./data_process/ytest.csv')
    except FileNotFoundError:
      print('No data file')
    
    y_test = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    testing = torch.cat((X,y_test), 1)

    try:
      model = IrisNetwork(3,0)
      model = torch.load('./model/model.pt')
    except FileNotFoundError:
      print('No model file')

    batch_size = int(load_file())

    

    inference(model, batch_size, testing)


if __name__ == "__main__":
    main()