"""
This script prepares the data, runs the training, and saves the model.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
from datetime import datetime

import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')

import mlflow
mlflow.autolog()

# Used classes and functions
# Tensor Transformer
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

def test_model(model, batch, test):
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



def train_model(model, n_epochs, batch_size, lr, train, val):
  train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  best_score = 0.0
  for epoch in range(n_epochs):
      for batch in train_loader:
          Xbatch = batch[:,:-1]
          ybatch = batch[:,-1].reshape(-1,1)
          y_pred = model(Xbatch)
          loss = loss_fn(y_pred, ybatch)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      _, score = test_model(model, batch_size, val)
      if best_score < score:
          best_score = score
          best_model = deepcopy(model)

  return best_model

# UPDATE Z MLFLOW 

# zapisać model wytrenowany w kontenerze, a potem na lokalnym dysku
# dodać logi: czas treningu, accuracy po każdym treningu, rozmiar zbioru i batch size przed treningiem

# Main execution ------------------------------------------------------------------------------------

# Import training datasets

X = pd.read_csv('./data_process/train.csv')
y = pd.read_csv('./data_process/ytrain.csv')

# Train-validation split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, shuffle = True)

X_train = prep.fit_transform(X_train)
X_val = prep.fit_transform(X_val)

y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1, 1)

training = torch.cat((X_train,y_train), 1)
validation = torch.cat((X_val,y_val), 1)















# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv('CONF_PATH') 

from utils import get_project_dir, configure_logging

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", 
                    help="Specify inference data file", 
                    default=conf['train']['table_name'])
parser.add_argument("--model_path", 
                    help="Specify the path for the output model")


class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df


class Training():
    def __init__(self) -> None:
        self.model = DecisionTreeClassifier(random_state=conf['general']['random_state'])

    def run_training(self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Running training...")
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        start_time = time.time()
        self.train(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.test(X_test, y_test)
        self.save(out_path)

    def data_split(self, df: pd.DataFrame, test_size: float = 0.33) -> tuple:
        logging.info("Splitting data into training and test sets...")
        return train_test_split(df[['x1','x2']], df['y'], test_size=test_size, 
                                random_state=conf['general']['random_state'])
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        logging.info("Training the model...")
        self.model.fit(X_train, y_train)

    def test(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        logging.info("Testing the model...")
        y_pred = self.model.predict(X_test)
        res = f1_score(y_test, y_pred)
        logging.info(f"f1_score: {res}")
        return res

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pickle')
        else:
            path = os.path.join(MODEL_DIR, path)

        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'])


if __name__ == "__main__":
    main()