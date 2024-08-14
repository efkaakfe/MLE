"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""
import os
import time 
import numpy as np
import pandas as pd
import logging 
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

'''from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler'''

sys.path.insert(1, os.getcwd())


from training.train import IrisNetwork, TensorTransformer, preprocessing

import warnings
warnings.filterwarnings('ignore')

# Setting the seed for reproductibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Logging 
logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

'''class IrisNetwork(nn.Module):
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
])'''

def load_file():
   try:
    with open('./model/batch_size.txt', 'r') as f:
       b = f.read()
       f.close
    return int(b)
   except FileNotFoundError:
    logging.info('No batch size file')
    sys.exit(1)
    
def save_results(true, pred):
   if not os.path.isdir('results'):
    os.mkdir('./results')
   rpath = os.path.join(os.getcwd(),'results','results.csv')
   r = pd.DataFrame({
     'Actual': true,
     'Predicted': pred
   })
   r.to_csv(rpath, index = False)
   logging.info(f'Results saved in: {rpath}')

def inference(model, batch, test):
  test_loader = DataLoader(test, batch_size = batch)
  true = test[:,-1].detach().numpy()
  test_loss = 0
  test_acc = 0
  loss_fn = nn.CrossEntropyLoss()
  prediction = np.array([])
  correct = 0
  l = 0
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
          correct += (y.detach().numpy()==pred).sum().item()
          l += batch.shape[0]
          prediction = np.concatenate((prediction,pred))

  e = time.time()

  test_loss /= l
  test_acc = correct/l

  logging.info(f"""Inference completed in {e-s} seconds.
               Accuracy score: {test_acc}.""")
  return true, prediction

def main():

    logging.info('Loading data')
    try: 
      X = pd.read_csv('./data_process/test.csv')
    except FileNotFoundError:
       logging.info('No data file')
       sys.exit(1)

    X = preprocessing(X)

    try: 
      y = pd.read_csv('./data_process/ytest.csv')
    except FileNotFoundError:
      logging.info('No data file')
      sys.exit(1)

    y_test = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

    testing = torch.cat((X,y_test), 1)

    logging.info('Loading model and batch size')

    try:
      model = IrisNetwork(3,0)
      model = torch.load('./model/model.pt')
    except FileNotFoundError:
      logging.info('No model file')
      sys.exit(1)

    batch_size = load_file()

    true, prediction = inference(model, batch_size, testing)
    save_results(true, prediction)

if __name__ == "__main__":
    main()
