
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


# In[2]:

df = pd.DataFrame()
df = pd.read_csv('/Users/apple/Documents/CSULA/machine_learning/CS5661/HW1/Cancer_small.csv')
df.head()


# In[3]:

print(list(df.columns.values))
feature_cols = ['Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']
X = df[feature_cols]
y = df['Malignant_Cancer']
print(X.shape)
print(y.shape)


# In[102]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
my_decisiontree = DecisionTreeClassifier(random_state=2)
my_decisiontree.fit(X_train, y_train)

print(X_train.shape)
print(y_train.shape)

y_predict = my_decisiontree.predict(X_test)
score = accuracy_score(y_test, y_predict)
print(score)


# In[64]:

#help(resample)
X_result=[]
feature_cols = ['Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']

bootstrap_size = int(0.8 * len(X_train['Clump_Thickness']))

df_predict = pd.DataFrame()

for i in range(19):
    ##sample from X_train
    X_sample = resample(X_train, n_samples = bootstrap_size , random_state=i , replace = True)
    #y_sample = resample(y_train, n_samples = bootstrap_size , random_state=i , replace = True)
    ###find the y value from y_train
    s_index = X_sample.index
    y_sample = y.iloc[s_index]
    Base_DecisionTree = DecisionTreeClassifier(random_state=2)
    Base_DecisionTree.fit(X_sample,y_sample)
    
    y_predict = Base_DecisionTree.predict(X_test)
    #score = accuracy_score(y_test, y_predict)
    #print(score)
    
    ##create matrix of predict results from 19 samples
    df_predict[i] = y_predict
    
print(df_predict.head())


# In[97]:

from statistics import mode
##voting
result = []
result = df_predict.values.tolist()
vote = []

for i in range(len(df_predict[0])):
    vote.append(mode(result[i]))
print(vote)


# In[98]:

##Test again with voting result
score = accuracy_score(y_test, vote)
print(score)


# In[103]:

from sklearn.ensemble import AdaBoostClassifier

#my_AdaBoost = AdaBoostClassifier(n_estimators = 19,random_state=2, base_estimator=my_decisiontree)
my_AdaBoost = AdaBoostClassifier(n_estimators = 19,random_state=2)
my_AdaBoost.fit(X_train, y_train)
y_predict_Adaboost = my_AdaBoost.predict(X_test)
score = accuracy_score(y_test, y_predict_Adaboost)
print(score)


# In[101]:

from sklearn.ensemble import RandomForestClassifier
my_RandomForest = RandomForestClassifier(n_estimators = 19, bootstrap = True, random_state=2)
my_RandomForest.fit(X_train, y_train)
y_predict_RandomForest = my_RandomForest.predict(X_test)
score = accuracy_score(y_test, y_predict_RandomForest)
print(score)


# In[ ]:



