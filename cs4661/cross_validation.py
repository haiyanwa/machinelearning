
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

credit_df = pd.read_csv('/Users/apple/Documents/CSULA/CS461/HW3/Credit.csv')
credit_df.head()


# In[3]:

##normalize function
def normalize(col):
    colmax = max(col)
    #print(colmax)
    col_s=[]
    for n in col:
        n_s = (n/colmax)
        col_s.append(n_s)
    return col_s

feature_cols = ['Income','Limit','Rating','Cards','Age','Education']

##new matrix with selected features which need to be normalized
credit_df_tmp = credit_df[feature_cols]

##print(max(credit_df['Balance']))

##new matrix for normalized training andy testing data
X= pd.DataFrame()

for f in feature_cols:
    normalized_col = normalize(credit_df_tmp[f])
    X[f] = normalized_col
X['Married'] =  credit_df['Married']   
print(X.head())

##target vector
y = credit_df['Balance']
#print(y.head())


# In[4]:

##split dataset to traning and testing data
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)


# In[5]:

from sklearn.linear_model import LinearRegression
my_linreg = LinearRegression()
my_linreg.fit(X_train, y_train)


# In[6]:

print(my_linreg.intercept_)
print(my_linreg.coef_)


# In[7]:

from IPython.display import Markdown
Markdown("The best feature is Rating and the least important feature is education")


# In[8]:

y_prediction = my_linreg.predict(X_test)
print(y_prediction)


# In[9]:

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
print(rmse)


# In[10]:

from sklearn.cross_validation import cross_val_score
mse_list = cross_val_score(my_linreg, X, y, cv=10, scoring='mean_squared_error')
print(mse_list)

mse_list_positive = -mse_list
rmse_list = np.sqrt(mse_list_positive)
print(rmse_list)


# In[11]:

print(rmse_list.mean())


# In[ ]:



