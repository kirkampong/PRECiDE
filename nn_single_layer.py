import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import Sequential
from keras.models import Input
from keras.layers import Dense
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import *
import sys
from pylab import *

dataset = pd.read_csv('data/african_crises.csv')

#for col in dataset.columns: 
#    print(col)

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

#is_crisis = 

X = dataset.loc[:,dataset.columns != 'banking_crisis']
y = dataset.loc[:, 'banking_crisis']

# Split the data into test train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Deep Neural Network Model
# - We can add more hidden layers to check whether it increases the accuracy or not.
# - The hyperparameters can be tuned to improve accuracy.

# Initialising the ANN
classifier = Sequential()
# Input layer
classifier.add(Dense(input_shape=(11,), units = 1, activation = 'sigmoid'))
#classifier.add(Dense(units = 8, activation = 'sigmoid')) # third (hidden) layer
# Output layer
#classifier.add(Dense(units = 1, activation = 'sigmoid'))
print(classifier.output_shape)
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the model
classifier.fit(X_train, y_train, epochs=200)


# Score model on train/test data
scores = classifier.evaluate(X_train, y_train)
print ("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = classifier.evaluate(X_test, y_test)
print ("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

y_pred = classifier.predict_classes(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

#Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['No Banking Crisis', 'Banking Crisis']
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the DNN Classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


