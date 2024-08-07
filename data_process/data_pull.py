# Importing required libraries
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import os

url = 'https://en.wikipedia.org/wiki/Iris_flower_data_set'

html = requests.get(url).content
df_list = pd.read_html(html)

data = pd.DataFrame(df_list[0])
data.set_index(data['Dataset order'], inplace = True)
data = data.drop('Dataset order', axis = 1)

train, test, ytrain, ytest = train_test_split(data.iloc[:,:-1], data.Species, test_size = 0.2, shuffle = True)

train = train.to_csv('./data_process/train.csv', index = False)
ytrain = ytrain.to_csv('./data_process/ytrain.csv', index = False)
test = test.to_csv('./data_process/test.csv', index = False)
ytest = ytest.to_csv('./data_process/yest.csv', index = False)