
# coding: utf-8

# In[2]:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from PIL import Image
get_ipython().magic('matplotlib inline')


# In[3]:

data_dir = "../../CS5661/HW2/"
label_file = data_dir + "label.csv"
df = pd.read_csv(label_file)



# In[16]:

import os.path
#print(label_df['name of the file'])
feature_data = []
i_dir = data_dir + "Digit/"

for i in range(len(df)):
    filename = df.loc[i,'name of the file']
    filepath = i_dir + str(filename) + ".jpg"
    if(os.path.isfile(filepath)):
        #print(filepath)
        img = Image.open(filepath)
        feature = (np.asarray(img.getdata())).tolist()
        if(i==1):
            print(feature)
        feature_data.append(feature)
        
df['feature'] =  feature_data  
        


# In[5]:

#Dataframe
X = df[['feature']]
y = df['digit']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=2)



# In[6]:

Xdata = []
ydata = []
for index, row in X.iterrows():
    Xdata.append(row['feature'])
    
for index, item in y.iteritems():
    ydata.append(item)

Xtrain = []
for index, row in X_train.iterrows():
    #print(index)
    Xtrain.append(row['feature'])
Xtest = []
ytrain = []
ytest = []

for index, row in X_test.iterrows():
    #print(index)
    Xtest.append(row['feature'])

for index, item in y_train.iteritems():
    #print(index, item)
    ytrain.append(item)

for index, item in y_test.iteritems():
    #print(index, item)
    ytest.append(item)
print(X_test[0:10])
print(y_test.index[0:10])


# In[7]:

my_ANN = MLPClassifier(hidden_layer_sizes=(80,), activation= 'logistic', 
                       solver='adam', alpha=1e-5, random_state=1, 
                       learning_rate_init = 0.002)
my_ANN.fit(Xtrain, ytrain)


# In[8]:

#print(my_ANN.coefs_)
#print(my_ANN.intercepts_)


# In[9]:

y_predict_ann = my_ANN.predict(Xtest)

print(ytest)
print(y_predict_ann.tolist())


# In[10]:

score_ann = accuracy_score(ytest, y_predict_ann)
print(score_ann)


# In[11]:

from sklearn import metrics

cm_ANN = metrics.confusion_matrix(y_test, y_predict_ann)

print("Confusion matrix:")
print(cm_ANN)


# In[12]:

mislabeled = []
for n in range(len(y_predict_ann)):
    if( y_predict_ann[n] != ytest[n]):
        #print("wrong prediction, index =", y_test.index[n])
        
        image_file = data_dir + "Digit/" + str(y_test.index[n]) + ".jpg"
        print(image_file)
        
        img = mpimg.imread(image_file)
        plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        print("label", ytest[n], "predict", y_predict_ann[n])


# In[13]:

file = "../../CS5661/HW2/Digit/1611.jpg"
img = mpimg.imread(file)
plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')


# In[14]:

file = "../../CS5661/HW2/Digit/123.jpg"
img = mpimg.imread(file)
plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')


# In[18]:

from sklearn.model_selection import GridSearchCV
neuron_number = [(i,) for i in range(50,200)]

param_grid = dict(hidden_layer_sizes = neuron_number)
print(param_grid,'\n')

my_ANN = MLPClassifier(activation='logistic', solver='adam', 
                                         alpha=1e-5, random_state=1, 
                                           learning_rate_init = 0.002)

grid = GridSearchCV(my_ANN, param_grid, cv=10, scoring='accuracy')
grid.fit(Xdata, ydata)
print("Best score", grid.best_score_)
print("Best param", grid.best_params_)


# In[ ]:




# In[ ]:



