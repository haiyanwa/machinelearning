
# coding: utf-8

# In[8]:

import cv2
import time
import re
import os.path
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import _pickle as cPickle


# In[9]:

start_time = time.time()

#imagedir_pos = "cyclist/48x96/48x96_Pos"
imagedir_pos = "/Volumes/Samsung_T5/DaimlerBenchmark/Data/TrainingData/Pedestrians/48x96/"
imagedir_non = "cyclist/48x96/48x96_Non"
savedir = "../"
flag = 0
sample_size_pos = 15660
sample_size_non = 9499
hist_size = 1980
feature_data = []
feature_data_non = []

 

## for 5763 x 1980 
for dirName, subdirList, fileList in os.walk(imagedir_pos):
    for fname in fileList:
        if(re.search(r"._pos*",fname)):
            continue
        filepath = dirName + "/" + fname
        image = cv2.imread(filepath,0)
        img = Image.open(filepath)
        hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
        hist = hog.compute(image)
        hist = hist.reshape((1,hist_size))
        
        if flag ==0: 
            print("image",image.shape)
            print("hist",hist.shape)
            flag = 1
        
        feature = np.asarray(hist)
        feature_data.append(feature[0])
        
        
print("feature_data[0]", (feature_data[0].shape)) 
print(len(feature_data))
current = time.time()

col_name = range(hist_size)
df1 = pd.DataFrame.from_records(feature_data, columns=col_name)

label_1s = np.ones(sample_size_pos, dtype=np.int)
df1['label'] = Series(label_1s, index=df1.index)

## for Non samples

for dirName, subdirList, fileList in os.walk(imagedir_non):
    for fname in fileList:
        filepath = dirName + "/" + fname
        image = cv2.imread(filepath,0)
        img = Image.open(filepath)
        hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
        hist = hog.compute(image)
        hist = hist.reshape((1,hist_size))
        feature = np.asarray(hist)
        feature_data_non.append(feature[0])

## for negative samples
non_index = range(sample_size_pos, sample_size_pos + sample_size_non )
df2 = pd.DataFrame.from_records(feature_data_non, index=non_index, columns=col_name)

label_0s = np.zeros(sample_size_non, dtype=np.int)
df2['label'] = Series(label_0s, index=df2.index)


print("finished in --- %s seconds ---" % (current - start_time))

df_matrix = pd.concat([df1, df2], axis=0)
#print(df_matrix.head(10))
#print(df_matrix.tail(10))


# In[6]:

feature_cols = list(range(1980))
X = df_matrix[feature_cols]
y = df_matrix['label']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=5)


# In[7]:

pca = decomposition.PCA(n_components=750)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#print(pca.explained_variance_ratio_)
#print(pca.explained_variance_ratio_.cumsum())


# In[19]:

my_SVM = SVC(C=1, kernel='rbf', gamma=0.0001,random_state=5)
my_SVM.fit(X_train_pca, y_train)
y_predict_svm= my_SVM.predict(X_test_pca)
score_svm = accuracy_score(y_test, y_predict_svm)
print(score_svm)


# In[20]:

with open('cyclist_classifier.pkl', 'wb') as fid:
    cPickle.dump((pca,my_SVM), fid)  


# In[21]:

with open('cyclist_classifier.pkl', 'rb') as fid:
    pca_file,cyclist_SVM_file = cPickle.load(fid)


# In[8]:

import re
import os.path
test_non = 1165
test_pos = 1030
hist_size = 1980
feature_testdata_non = []
feature_testdata_pos = []

##negative test data
testimagedir0 = 'cyclist/48x96_test/image_test0'
for dirName, subdirList, fileList in os.walk(testimagedir0):
    for fname in fileList:
        if(not re.search(r".png",fname)):
            continue
        filepath = dirName + "/" + fname
        image = cv2.imread(filepath,0)
        hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
        hist = hog.compute(image)
        hist = hist.reshape((1,hist_size))
        feature = np.asarray(hist)
        feature_testdata_non.append(feature[0])
        
col_name = range(hist_size)
df_non = pd.DataFrame.from_records(feature_testdata_non, columns=col_name)

label_0s = np.zeros(test_non, dtype=np.int)
df_non['label'] = Series(label_0s, index=df_non.index)

##positive test data
testimagedir1 = 'cyclist/48x96_test/image_test1'
for dirName, subdirList, fileList in os.walk(testimagedir1):
    for fname in fileList:
        if(not re.search(r".png",fname)):
            continue
        filepath = dirName + "/" + fname
        image = cv2.imread(filepath,0)
        hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
        hist = hog.compute(image)
        hist = hist.reshape((1,hist_size))
        feature = np.asarray(hist)
        feature_testdata_pos.append(feature[0])
        
pos_index = range(test_non, test_non + test_pos)        
df_pos = pd.DataFrame.from_records(feature_testdata_pos, index=pos_index, columns=col_name)

label_1s = np.ones(test_pos, dtype=np.int)
df_pos['label'] = Series(label_1s, index=df_pos.index)

df_test_matrix = pd.concat([df_non, df_pos], axis=0)
print(df_test_matrix.head(10))
print(df_test_matrix.tail(10))


# In[57]:

feature_cols = list(range(1980))
X_test1 = df_test_matrix[feature_cols]
y_test1 = df_test_matrix['label']


# In[58]:

with open('cyclist_classifier.pkl', 'rb') as fid:
    pca_file,cyclist_SVM_file = cPickle.load(fid)
    
X_test1_pca = pca_file.transform(X_test1)
y_test1_predict= cyclist_SVM_file.predict(X_test1_pca)

score_test1 = accuracy_score(y_test1, y_test1_predict)
print(score_test1)


# In[59]:

false_detection = []
for i in range(len(y_test1)):
    if y_test1_predict[i] != y_test1[i]:
        false_detection.append(i)


# In[60]:

##for retrain with pedestrians as Non samples
df_retrain_matrix = pd.concat([df_matrix, df_test_matrix], axis=0)
df_retrain_matrix.shape


# In[61]:

X_r = df_retrain_matrix[feature_cols]
y_r = df_retrain_matrix['label']
#X_train_r, X_test_r, y_train_r, y_test_r= train_test_split(X_r, y_r, test_size=0.1, random_state=5)


# In[62]:

pca_r = decomposition.PCA(n_components=750)
#pca_r.fit(X_train_r)
#X_train_pca_r = pca_r.transform(X_train_r)
#X_test_pca_r = pca_r.transform(X_test_r)
pca_r.fit(X_r)
X_train_pca_r = pca_r.transform(X_r)
print(pca_r.explained_variance_ratio_)
print(pca_r.explained_variance_ratio_.cumsum())


# In[63]:

my_SVM_retrain = SVC(C=1, kernel='rbf', gamma=0.0001,random_state=5)
#my_SVM_retrain.fit(X_train_pca_r, y_train_r)
my_SVM_retrain.fit(X_train_pca_r, y_r)

#y_predict_svm_r= my_SVM_retrain.predict(X_test_pca_r)
#score_svm = accuracy_score(y_test_r, y_predict_svm_r)
#print(score_svm)
#output: 0.935280641466


# In[54]:

print(y_test_r)
y_test_r_list = y_test_r.tolist()
false_detection = []
for k in range(len(y_test_r_list)):
    if y_predict_svm_r[k] != y_test_r_list[k]:
        print(k)


# In[65]:

##test on separate test set
X_test1_pca_r = pca_r.transform(X_test1)
y_test1_predict_r= cyclist_SVM_file.predict(X_test1_pca_r)

score_test_r = accuracy_score(y_test1, y_test1_predict_r)
print(score_test_r)


# In[28]:

with open('cyclist_classifier_retrain.pkl', 'wb') as fid:
    cPickle.dump((pca_r,my_SVM_retrain), fid)


# In[81]:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
images = []

count=0
testimagedir0 = 'cyclist/48x96_test/image_test0'
for dirName, subdirList, fileList in os.walk(testimagedir0):
    for fname in fileList:
        filepath = dirName + "/" + fname
        #image = cv2.imread(filepath,0)
        image = mpimg.imread(filepath)
        image_arr = np.asarray(image)
        images.append(image_arr)

testimagedir1 = 'cyclist/48x96_test/image_test1'
for dirName, subdirList, fileList in os.walk(testimagedir1):
    for fname in fileList:
        filepath = dirName + "/" + fname
        image = mpimg.imread(filepath)
        image_arr = np.asarray(image)
        images.append(image_arr)

print(images[0])
get_ipython().magic('matplotlib inline')
plt.imshow(images[972],cmap=plt.cm.gray)


# In[ ]:

X_pca = np.concatenate([X_train_pca, X_test_pca])
y_m = np.concatenate([y_train, y_test])

param_grid = {'C': [0.1, 1,10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
             'gamma': [0.0001, 0.0005],}
grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid,cv=10, scoring='accuracy')
grid.fit(X_pca, y_m)
print("Best score", grid.best_score_)
print("Best param", grid.best_params_)


# In[ ]:



