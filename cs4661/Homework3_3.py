
# coding: utf-8

# In[8]:

# Importing libraries and packages:

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[9]:

#Problem a
cancer_df = pd.read_csv('Heart_short.csv')

# checking the dataset by printing every 10 lines:
cancer_df.head()


# In[26]:

def categorical_to_numeric(x):
    if x == 'No':
        return 0
    elif x == 'Yes':
        return 1
cancer_df['AHD'] = cancer_df['AHD'].apply(categorical_to_numeric)


# In[27]:

#Problem b
feature_cols = ['Age','RestBP','Chol','RestECG','MaxHR','Oldpeak']

# use the above list to select the features from the original DataFrame
X = cancer_df[feature_cols] 
#normalization
X = preprocessing.scale(X)
# select a Series of labels (the last column) from the DataFrame
y = cancer_df['AHD']

# print the first 5 rows
print(X)
print(y.head())


# In[28]:

#Problem c
# Randomly splitting the original dataset into training set and testing set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# print the size of the traning set:
print(X_train.shape)
print(y_train.shape)

# print the size of the testing set:
print(X_test.shape)
print(y_test.shape)


# In[29]:

#Problem d
# "my_logreg" is instantiated as an "object" of LogisticRegression "class". 
my_logreg = LogisticRegression()
# Training ONLY on the training set:
my_logreg.fit(X_train, y_train)
# Testing on the testing set:
y_predict_lr = my_logreg.predict(X_test)
print(y_predict_lr)
# We can now compare the "predicted labels" for the Testing Set with its "actual labels" to evaluate the accuracy 

score_lr = accuracy_score(y_test, y_predict_lr)

print(score_lr)


# In[30]:

#Problem e
# Estimating the probability (likelihood) of Each Label: 
y_predict_prob_lr = my_logreg.predict_proba(X_test)

# This line prints the "estimated likelihood of label=1" for the testing set:
print(y_predict_prob_lr[:,1])


# In[31]:

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob_lr[:,1], pos_label=1)

print(fpr)
print(tpr)
# AUC:
AUC = metrics.auc(fpr, tpr)
print(AUC)


# In[32]:

# Importing the "pyplot" package of "matplotlib" library of python to generate 
# graphs and plot curves:
import matplotlib.pyplot as plt

# The following line will tell Jupyter Notebook to keep the figures inside the explorer page 
# rather than openng a new figure window:
get_ipython().magic('matplotlib inline')

plt.figure()

# Roc Curve:
plt.plot(fpr, tpr, color='red', lw=2, 
         label='ROC Curve (area = %0.2f)' % AUC)

# Random Guess line:
plt.plot([0, 1], [0, 1], color='blue', lw=1, linestyle='--')

# Defining The Range of X-Axis and Y-Axis:
plt.xlim([-0.005, 1.005])
plt.ylim([0.0, 1.01])

# Labels, Title, Legend:
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.show()


# In[ ]:



