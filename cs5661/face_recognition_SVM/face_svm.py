
# coding: utf-8

# In[1]:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# In[2]:

data_dir = "../../CS5661/HW3/Face/"
label_file = data_dir + "label.csv"
df = pd.read_csv(label_file)
print(df.head())


# In[3]:

get_ipython().magic('matplotlib inline')
file_name = data_dir + "images/" + "0.jpg"
image = mpimg.imread(file_name)
plt.imshow(image, cmap=plt.cm.gray)


# In[4]:

image_dir = data_dir + "images/"
feature_data = []
for i in range(len(df)):
    filepath = image_dir + str(i) + ".jpg"
    
    img = Image.open(filepath)
    feature = np.asarray(img.getdata())
    feature_data.append(feature)
print(len(feature_data))
print(len(feature_data[0]))


# In[6]:

feature_data_scaled = preprocessing.scale(feature_data)
print(feature_data_scaled.shape)
col_name = range(4096)
df1 = pd.DataFrame.from_records(feature_data_scaled, columns=col_name)


# In[6]:

df_matrix = pd.concat([df, df1], axis=1)
print(df_matrix.head())


# In[7]:

feature_cols = list(range(4096))
X = df_matrix[feature_cols]
y = df_matrix['Label']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25, random_state=5)
print(X_train.shape)
print(y_train.shape)


# In[8]:

pca = decomposition.PCA(n_components=50)
pca.fit(X)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)
print(X_test_pca.shape)
print(type(X_train_pca))


# In[9]:

my_SVM = SVC(C=1, kernel='rbf', gamma=0.0005,random_state=1)
my_SVM.fit(X_train_pca, y_train)
y_predict_svm= my_SVM.predict(X_test_pca)
score_svm = accuracy_score(y_test, y_predict_svm)
print(score_svm)


# In[10]:

cm_SVM = metrics.confusion_matrix(y_test, y_predict_svm)
print("Confusion matrix:")
print(cm_SVM)


# In[23]:

X_pca = np.concatenate([X_train_pca, X_test_pca])
y_m = np.concatenate([y_train, y_test])

X_pca.shape


# In[24]:

param_grid = {'C': [0.1, 1,10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
             'gamma': [0.0001, 0.0005],}

#grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, scoring='accuracy')
#grid.fit(X_train_pca, y_train)
grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid,cv=10, scoring='accuracy')
grid.fit(X_pca, y_m)
print("Best score", grid.best_score_)
print("Best param", grid.best_params_)


# In[ ]:



