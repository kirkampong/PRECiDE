import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

##Some configuration settings
pd.set_option("display.max_columns", 100)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#import the data
dataframe = pd.read_csv('african_crises.csv', index_col = 0) #index col neccessary?
dataframe.head()

print(dataframe.shape) #(1059, 13)


# Plot total number of cases by country
sns.countplot(x='country', data=dataframe, palette ='hls')
plt.xticks(rotation=90)
plt.show()

# Plot counts of 'x'_crisis, where 'x' can equals: systemic, currency, inflation, branking.
sns.countplot(x='banking_crisis', data=dataframe, palette ='hls')
plt.xticks(rotation=90)
plt.show()
print(dataframe['banking_crisis'].value_counts()) #print counts

#------------------------------------------------------------------------------------------
#					PRE-PROCESSING
#------------------------------------------------------------------------------------------

# Drop the categorical variables which are not useful to the dataset for logistic regression
dataframe = dataframe.drop(['cc3', 'country', 'year'], axis =1)
dataframe.head()

# drop this column since it is not informative (jsutify with plot later)
dataframe = dataframe.drop(['gdp_weighted_default'], axis =1)
dataframe.head()

# Define ouput column y and drop from dataset
y = dataframe[['banking_crisis']]
y = pd.get_dummies(dataframe['banking_crisis'],drop_first=True)
dataframe = dataframe.drop(['banking_crisis'], axis =1)
dataframe.head()


#------------------------------------------------------------------------------------------
#				  TRAINING
#------------------------------------------------------------------------------------------

# split the data into test train sets
from sklearn.model_selection import train_test_split
# create training and testing vars
X_train, X_test, Y_train, Y_test = train_test_split(dataframe, y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
# train
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, Y_train)
# predict
Predictions = logmodel.predict(X_test)
print("*********logmodel.coef_[0]**********")
print(logmodel.coef_[0])
print("************************************")


#------------------------------------------------------------------------------------------
#			      REPORT
#------------------------------------------------------------------------------------------
from sklearn.metrics import classification_report
print("=====classification_report=====")
print(classification_report(Y_test,Predictions))
# confusion matrix
print("=====confusion_matrix=====")
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Predictions))

# calculate the fpr and tpr for all thresholds of the classification
import sklearn.metrics as metrics
probs = logmodel.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)



#------------------------------------------------------------------------------------------
#			      DISPLAY
#------------------------------------------------------------------------------------------


# method I: plt
import matplotlib.pyplot as plt
plt.title('Flase Positives /True Positives')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
