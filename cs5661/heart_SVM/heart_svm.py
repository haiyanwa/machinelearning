
# coding: utf-8

# In[129]:

from IPython.display import HTML
import numpy as np
import pandas as pd
import html5lib
import matplotlib as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn import decomposition
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import requests
from bs4 import BeautifulSoup
import csv
import re # regular expressions library

get_ipython().magic('matplotlib inline')



# In[72]:

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data'

webpage = requests.get(data_url)

webcontents = webpage.text # this returns the webpage html content

#print(data)
list = []
list=re.split(r'\n', webcontents)
datalist = []
data = []
for i in range(len(list)):
    datalist = re.split(r'\s+', list[i])
    data.append(datalist)
    #print(datalist)

#print(len(data[0]))


# In[68]:

name_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names'
webpage = requests.get(name_url)
contents = webpage.text
#print(contents)


# In[44]:

#-- 1. #3  (age) 
matched = []
matched = re.findall(r'--\s\d+\.\s\#\d+\s+\(\w+\)', contents)
taglist = []
for i in range(len(matched)):
    m = re.match(r"--\s\d+\.\s\#\d+\s+\((\w+)\)", matched[i])
    print(i,m.group(1) )
    taglist.append(m.group(1))


# In[63]:

print(type(taglist))
df = pd.DataFrame.from_records(data, columns=taglist)
print(df.head())


# In[66]:

##drop na
df = df.dropna(axis=0, how='any')
df.shape


# In[114]:

feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[feature_cols]
X = preprocessing.scale(X)
y = df['num']
#print(X[0:10])


# In[115]:

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=5)


# In[137]:

##pick 11 because the ratio is over 95%
pca = decomposition.PCA(n_components=11)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())


# In[138]:

my_SVM = SVC(C=1, kernel='rbf', gamma=0.1,random_state=5)
my_SVM.fit(X_train_pca, y_train)
y_predict_svm= my_SVM.predict(X_test_pca)
score_svm = accuracy_score(y_test, y_predict_svm)
print(score_svm)


# In[132]:

my_ANN = MLPClassifier(hidden_layer_sizes=(100,20), activation= 'logistic', 
                       solver='adam', alpha=1e-5, random_state=5, 
                       learning_rate_init = 0.02)
my_ANN.fit(X_train, y_train)


# In[131]:

y_predict_ann = my_ANN.predict(X_test)
score_ann = accuracy_score(y_test, y_predict_ann)
print(score_ann)


# In[ ]:



