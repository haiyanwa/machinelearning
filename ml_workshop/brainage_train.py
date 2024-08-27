#!/usr/bin/env python3

import numpy as np
import pandas as pd
import h5py
import sklearn
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, MaxPooling2D, GlobalAveragePooling2D, Flatten
import pickle
from brainage_models import small_3d_model, densenet 

data_path = "/Users/hwang_1/Documents/scripts/python/hoffman/data/brainage"
df = pd.read_csv(os.path.join(data_path, "train_brainage.csv"))

##number of data of senior
#df[df['age_years'] > 40].count()


##pick 4 data from age > 40 and 8 from age < 40 as testing data
##Use the rest for training
df_test_s = df[df['age_years'] > 40].sample(4, replace=False)

df_test_j = df[df['age_years'] < 40].sample(8, replace=False)

test_df = pd.concat([df_test_s, df_test_j])
train_df = df.drop(test_df.index)

def read_scan(path, in_filename, folder='train'):
    full_scan_path = os.path.join(data_path,folder, in_filename)
    # load the image using hdf5
    with h5py.File(full_scan_path, 'r') as h:
        return h['image'][:][:, :, :, 0] # we read the data from the file

##data generator
def data_gen(in_df, batch_size=8, model="2d"):
    """Generate image and age label data in batches"""
    while True:
        images, age = [], []
        balanced_sample_df = in_df.groupby(in_df['age_years']<40).apply(lambda x: x.sample(batch_size//2)).reset_index(drop=True)
    
        for _, c_row in balanced_sample_df.iterrows():
            age += [c_row['age_years']]
            img = read_scan(data_path, c_row['h5_path'])
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean)/ std
            images += [img]
            
        if model == "3d":
            yield np.expand_dims(np.stack(images, 0), -1), np.expand_dims(np.stack(age), -1)
        else:
            yield np.stack(images, 0), np.stack(age)

##for 3d model
train_gen_3d = data_gen(train_df, model="3d")
test_gen_3d = data_gen(test_df, model="3d")

##for 2d model
train_gen = data_gen(train_df)
test_gen= data_gen(test_df)

X, y = next(train_gen_3d)
print(X.shape, y.shape)


cnn3d_model = small_3d_model()

train_len = len(train_df_3d)
test_len = len(test_df_3d)
print(train_len, test_len)

checkpoint_path = "checkpoints/cnn3d_checkpoint"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                               save_weights_only=True, # save model weights
                               monitor="val_mae", 
                               save_best_only=True # save the model weights which score the best validation accuracy
                               )
checkpoint_callback2 = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                                           patience=3, min_lr=0.00001, verbose=1)


cnn3d_model_history = cnn3d_model.fit(train_gen_3d,
                 epochs = 30,
                 steps_per_epoch=12, 
                 validation_data = test_gen_3d,
                 validation_steps = 2,
                 verbose=True, 
                 callbacks = [checkpoint_callback]
                 )

cnn3d_model.save("model/cnn3d_model.keras")
#new_model = tf.keras.models.load_model('model/cnn3d_model.keras')

history_ccn3d_model_path = "history/cnn3d_model"
with open(history_ccn3d_model_path, 'wb') as file_open:
    pickle.dump(history_ccn3d_model_path, file_open)
 
   
for index, row in test_df.iterrows():    
    age = row.age_years
    test_data = read_scan(data_path, row.h5_path)
    mean = np.mean(test_data)
    std = np.std(test_data)
    test_data = (test_data - mean)/std
    test_data = np.expand_dims(test_data, -1)
    test_data = np.expand_dims(test_data, 0)
    print(test_data.shape)
    y_pred = cnn3d_model.predict(test_data)
    print(y_pred, age)
