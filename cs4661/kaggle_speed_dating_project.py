
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
#pd.__version__


# In[4]:

Speed_Dating_df = pd.DataFrame()
Speed_Dating_df = pd.read_csv('Speed_Dating_Data.csv',encoding = "ISO-8859-1")
pd.set_option('display.max_column',40)
pd.set_option('display.max_row',20)
Speed_Dating_df


# In[15]:

#Speed_Dating_Data.csv  speedating.csv
Speed_Dating_df = pd.DataFrame()
Speed_Dating_df = pd.read_csv('Speed_Dating_Data.csv',encoding = "ISO-8859-1")
print(Speed_Dating_df.shape)

import bottleneck as bn
def mean_age_of_wave(x):
    age = []
    for i in range(len(Speed_Dating_df['age'])):
        if Speed_Dating_df['wave'][i] == x:
            age.append(Speed_Dating_df['age'][i])
    median_age = round(bn.nanmean(age),2)
    return median_age

##samerace
index = Speed_Dating_df['samerace'].index[Speed_Dating_df['samerace'].apply(np.isnan)]
print("missing samerace",len(index))

##age
index = Speed_Dating_df['age'].index[Speed_Dating_df['age'].apply(np.isnan)]

#Speed_Dating_df.loc[index,'age'] = 25
for i in index:
    Speed_Dating_df.loc[i,'age'] = mean_age_of_wave(Speed_Dating_df['wave'][i])

index = Speed_Dating_df['age'].index[Speed_Dating_df['age'].apply(np.isnan)]
print("missing age",len(index))


# In[16]:

##age_o
index = Speed_Dating_df['age_o'].index[Speed_Dating_df['age_o'].apply(np.isnan)]

for i in index:
    Speed_Dating_df.loc[i,'age_o'] = mean_age_of_wave(Speed_Dating_df['wave'][i])
index_o = Speed_Dating_df['age_o'].index[Speed_Dating_df['age_o'].apply(np.isnan)]
print("missing age_o",len(index_o))


# In[17]:

###imprace shows how important the perticipiant think the importance of being the same race
###only 10 data are missing, so fill up NAN as 0
Speed_Dating_df['imprace'].fillna(value=0, inplace=True)
##same for imprelig since only 16 data are missing
Speed_Dating_df['imprelig'].fillna(value=0, inplace=True)


# In[46]:

print(Speed_Dating_df.shape)
##Drop NAN from the following 7 columns
Speed_Dating_df = Speed_Dating_df.dropna(subset=['attr','sinc','intel','fun','amb','shar','like','dec'],how='any')
print(Speed_Dating_df.shape)


# In[47]:

##Double check any NAN
index_attr = Speed_Dating_df['attr'].index[Speed_Dating_df['attr'].apply(np.isnan)]
print(len(index_attr))
index_sinc = Speed_Dating_df['sinc'].index[Speed_Dating_df['sinc'].apply(np.isnan)]
print(len(index_sinc))


# In[48]:

X= pd.DataFrame()

y = Speed_Dating_df['match']
##
#feature_cols =['iid','gender','wave','samerace','age','age_o','imprace','imprelig','career_c','attr','sinc','intel','fun','amb','shar','like']
feature_cols =['attr','sinc','intel','fun','amb','shar','like']
##wave: round
##age_o: age of partner

def max_of_wave(f,x):
    data = []
    for i in range(len(Speed_Dating_df[f])):
        if Speed_Dating_df['wave'][i] == x:
            data.append(Speed_Dating_df[f][i])
    max_data = round(bn.nanmax(data),2)
    return max_data

X = Speed_Dating_df[feature_cols]

#onehot_gender = pd.get_dummies(Speed_Dating_df['gender'])
#onehot_gender[0]

#X = pd.concat([X, pd.get_dummies(Speed_Dating_df['gender'])], axis=1)
#X.rename(columns={0:"f", 1: "m"}, inplace=True)
#X = pd.concat([X, pd.get_dummies(Speed_Dating_df['samerace'])], axis=1)
#X.rename(columns={0:"samerace", 1: "notsamerace"}, inplace=True)


X = pd.concat([X, (Speed_Dating_df['age'] - Speed_Dating_df['age_o']).abs()], axis=1)
X.rename(columns={0:"age_diff"}, inplace=True)
#X = pd.concat([X, pd.get_dummies(Speed_Dating_df['career_c'])], axis=1)

##change samerace: 1 -> same -1-> not same
Speed_Dating_df.loc[Speed_Dating_df['samerace'] == 0,'samerace'] = -1 

##new race column by 
##row_index =  Speed_Dating_df[Speed_Dating_df['samerace'] == -1].index.tolist()
X = pd.concat([X,(Speed_Dating_df['samerace'] * Speed_Dating_df['imprace'])],axis=1)

X.rename(columns={0:"imprace"}, inplace=True)
print(X.head())


# In[49]:

##normalize
from sklearn import preprocessing
X = preprocessing.scale(X)


# In[50]:

###Use Linear regression to find out which factor is more important
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
my_linreg = LinearRegression()

my_linreg.fit(X_train, y_train)
print("Theta0: ", my_linreg.intercept_)
print("coef of the features:\n ", my_linreg.coef_)


# In[51]:

y_prediction = my_linreg.predict(X_test)
print(y_prediction)


# In[52]:

##RMSE
from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
print(rmse)


# In[53]:

## Use cross validation
from sklearn.cross_validation import cross_val_score
mse_list = cross_val_score(my_linreg, X, y, cv=10, scoring='mean_squared_error')
mse_list_positive = -mse_list
rmse_list = np.sqrt(mse_list_positive)
print(rmse_list)
print(rmse_list.mean())


# In[78]:

###Can these features explain the decisions or not?
###Use dec (decision) instead of match
y_dec = Speed_Dating_df['dec']
feature_cols =['attr','sinc','intel','fun','amb','shar','like','imprace','imprelig']
X_dec = pd.DataFrame()
X_dec = Speed_Dating_df[feature_cols]

X_dec = pd.concat([X_dec, pd.get_dummies(Speed_Dating_df['gender'])], axis=1)
X_dec.rename(columns={0:"f", 1: "m"}, inplace=True)

X_dec = pd.concat([X_dec, (Speed_Dating_df['age'] - Speed_Dating_df['age_o']).abs()], axis=1)
X_dec.rename(columns={0:"age_diff"}, inplace=True)

print(X_dec.head())

X_dec = preprocessing.scale(X_dec)

X_dec_train, X_dec_test, y_dec_train, y_dec_test = train_test_split(X_dec, y_dec, test_size=0.3, random_state=2)
my_linreg = LinearRegression()

print(X_dec_train.shape)
print(y_dec_train.shape)
print(X_dec_test.shape)
print(y_dec_test.shape)

my_linreg.fit(X_dec_train, y_dec_train)
print("Theta0: ", my_linreg.intercept_)
print("coef: ", my_linreg.coef_)
y_prediction = my_linreg.predict(X_dec_test)
print(y_prediction)

mse = metrics.mean_squared_error(y_dec_test, y_prediction)
rmse = np.sqrt(mse)
print(rmse)
'''
When using "dec" instead of "match" as the target, Error rate actually got bigger
'''


# In[63]:

##KNN
k = 10
knn = KNeighborsClassifier(n_neighbors=k)
accuracy_list = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(accuracy_list)
accuracy_cv = accuracy_list.mean()
print("KNN accuracy rate mean: ", accuracy_cv)


# In[64]:

##decision tree
my_decisiontree = DecisionTreeClassifier()
accuracy_list = cross_val_score(my_decisiontree, X, y, cv=10, scoring='accuracy')
print(accuracy_list)
accuracy_cv = accuracy_list.mean()
print("Decision tree accuracy rate mean: ", accuracy_cv)


# In[65]:

##Logistic regression
my_logreg = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
my_logreg.fit(X_train, y_train)
y_predict_lr = my_logreg.predict(X_test)
from sklearn.metrics import accuracy_score
score_lr = accuracy_score(y_test, y_predict_lr)
print("Logistic Accuracy score", score_lr)

print("Logistic Cross validation score:")
accuracy_list = cross_val_score(my_logreg, X, y, cv=10, scoring='accuracy')
print(accuracy_list)
accuracy_cv = accuracy_list.mean()
print("Logistic Cross validation accuracy mean:", accuracy_cv)

y_predict_prob_lr = my_logreg.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob_lr[:,1], pos_label=1)
AUC = metrics.auc(fpr, tpr)
print("AUC", AUC)

import matplotlib.pyplot as plt
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


# In[67]:

###Differece between female and male?
female_row_index =  Speed_Dating_df[Speed_Dating_df['gender'] == 0].index.tolist()
male_row_index =  Speed_Dating_df[Speed_Dating_df['gender'] == 1].index.tolist()
X_f = Speed_Dating_df.loc[female_row_index,['attr','sinc','intel','fun','amb','shar','like']]
X_m = Speed_Dating_df.loc[male_row_index,['attr','sinc','intel','fun','amb','shar','like']]


y_f = Speed_Dating_df.loc[female_row_index,'dec']
y_m = Speed_Dating_df.loc[male_row_index,'dec']
#y_f = Speed_Dating_df.loc[female_row_index,'dec']
#y_m = Speed_Dating_df.loc[male_row_index,'dec']

X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_f, y_f, test_size=0.2, random_state=2)

f_linreg = LinearRegression()

f_linreg.fit(X_f_train, y_f_train)
print("female ", f_linreg.intercept_)
print("female ", f_linreg.coef_)


X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(X_m, y_m, test_size=0.2, random_state=2)

m_linreg = LinearRegression()

m_linreg.fit(X_m_train, y_m_train)
print("male ", m_linreg.intercept_)
print("male ", m_linreg.coef_)


##KNN female
k = 10
knn = KNeighborsClassifier(n_neighbors=k)
accuracy_list = cross_val_score(knn, X_f, y_f, cv=10, scoring='accuracy')
#print("KNN accuracy list for female: ", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("------------------------------------------")
print("KNN for female", accuracy_cv)

##KNN male
accuracy_list = cross_val_score(knn, X_m, y_m, cv=10, scoring='accuracy')
#print("KNN accuracy list for female: ", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("KNN for male", accuracy_cv)
print("------------------------------------------")

##decision tree for female
my_decisiontree = DecisionTreeClassifier()
accuracy_list = cross_val_score(my_decisiontree, X_f, y_f, cv=10, scoring='accuracy')
#print("Decision Tree accuracy list for female: ", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("Decision Tree mean for female: ", accuracy_cv)
##decision tree for male
my_decisiontree = DecisionTreeClassifier()
accuracy_list = cross_val_score(my_decisiontree, X_m, y_m, cv=10, scoring='accuracy')
#print("Decision Tree accuracy list for male: ", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("Decision Tree mean for male: ", accuracy_cv)
print("------------------------------------------")

##Logistic regression for female
my_logreg = LogisticRegression()
accuracy_list = cross_val_score(my_logreg, X_f, y_f, cv=10, scoring='accuracy')
#print("Logistic accuracy list for female: ", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("Logistic accuracy mean for female: ", accuracy_cv)

##Logistic regression for male
my_logreg = LogisticRegression()
accuracy_list = cross_val_score(my_logreg, X_m, y_m, cv=10, scoring='accuracy')
#print("Logistic accuracy list for male: ", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("Logistic accuracy mean for male: ", accuracy_cv)


# In[41]:

print ("Different between self evaluation and partner evaluation?")
SData = pd.DataFrame()
SData = Speed_Dating_df[['attr3_1','sinc3_1','intel3_1','fun3_1','amb3_1','attr_o','sinc_o','intel_o','fun_o','amb_o','shar','shar_o','like','like_o','match']]
SData.head()


# In[42]:

SData = SData.dropna(subset=['attr3_1','sinc3_1','intel3_1','fun3_1','amb3_1','attr_o','sinc_o','intel_o','fun_o','amb_o','shar','shar_o','like','like_o','match'],how='any')


# In[43]:

X2 = pd.DataFrame()
X2 = SData[['attr3_1','sinc3_1','intel3_1','fun3_1','amb3_1',
           'attr_o','sinc_o','intel_o','fun_o','amb_o',
           'shar','shar_o','like','like_o']]
print(X2.shape)
y2 = SData['match']
print(y2.shape)


# In[44]:

##Logistic regression
my_logreg = LogisticRegression()
accuracy_list = cross_val_score(my_logreg, X2, y2, cv=10, scoring='accuracy')
print("Logistic cross validation: \n", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("Logistic cross validation mean: \n", accuracy_cv)

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=2)
my_logreg.fit(X_train, y_train)
y_predict_lr = my_logreg.predict(X_test)
score_lr = accuracy_score(y_test, y_predict_lr)
print("Logistic Accuracy score", score_lr)

y_predict_prob_lr = my_logreg.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob_lr[:,1], pos_label=1)
AUC = metrics.auc(fpr, tpr)
print("AUC", AUC)

import matplotlib.pyplot as plt
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


# In[45]:

##KNN
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
accuracy_list = cross_val_score(knn, X2, y2, cv=10, scoring='accuracy')
print("KNN", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("KNN accuracy mean",accuracy_cv)


# In[46]:

##decision tree
my_decisiontree = DecisionTreeClassifier()
accuracy_list = cross_val_score(my_decisiontree, X2, y2, cv=10, scoring='accuracy')
print("Decision tree", accuracy_list)
accuracy_cv = accuracy_list.mean()
print("Decision tree accuracy mean", accuracy_cv)


# In[71]:

'''
Accuracy improved in this case than using only the one-sided evaluation model
'''


# In[ ]:



