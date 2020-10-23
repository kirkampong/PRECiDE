import pandas as pd
import numpy as np
from sklearn import preprocessing

dataset = pd.read_csv('african_crises.csv')

for col in dataset.columns: 
    print(col)

#Preprocessing data
dataset['banking_crisis'] = dataset['banking_crisis'].replace('crisis',np.nan)
dataset['banking_crisis'] = dataset['banking_crisis'].fillna(1)
dataset['banking_crisis'] = dataset['banking_crisis'].replace('no_crisis',np.nan)
dataset['banking_crisis'] = dataset['banking_crisis'].fillna(0)
dataset.drop(['cc3','country'], axis=1, inplace=True)

#Feature scaling
dataset_scaled = preprocessing.scale(dataset)
dataset_scaled = pd.DataFrame(dataset_scaled, columns=dataset.columns)
dataset_scaled['banking_crisis'] = dataset['banking_crisis']
dataset = dataset_scaled