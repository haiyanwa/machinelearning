#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.layers import Input, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, AveragePooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import Model
from keras.layers import concatenate

def densenet(height, width, depth, num_init_features=64, growth_rate=32, drop_rate=0, block_config = (6, 12, 24, 16),bn_size=4,chan_dim=-1): 
    def layer0(inputs, chan_dim=-1):
        x = tf.keras.layers.ZeroPadding2D(padding=3)(inputs)
        x = Conv2D(num_init_features, 7, strides=(2,2), use_bias=False)(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation("relu")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=1)(x)
        output = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return output

    def trans_layer(inputs, num_features, chan_dim=-1):
        x = BatchNormalization(axis=chan_dim)(inputs)
        x = Activation("relu")(x)
        ##use 1x1 conv layer to shink output channels to half
        x = Conv2D(num_features, (1,1), strides=1, use_bias=False)(x)
        ##decrease image dimensions
        output = MaxPooling2D((2,2), (2,2))(x)
        return output

    def denseLayer(inputs, growth_rate=32, bn_size=4, drop_rate=0, chan_dim=-1):
        concated = tf.concat(inputs, -1)
        ##bottle neck
        norm1 = BatchNormalization(axis=chan_dim)(concated)
        relu1 = Activation("relu")(norm1)
        conv1 = Conv2D(bn_size * growth_rate, 1, strides=(1,1), use_bias=False)(relu1)

        norm2 = tf.keras.layers.BatchNormalization(axis=chan_dim)(conv1)
        relu2 = Activation("relu")(norm2)
        output = Conv2D(growth_rate, 3, strides=(1,1), padding="same", use_bias=False)(relu2)
        return output

        features = [init_features]
        for i in range(num_layers):
            features_new = denseLayer(features, growth_rate, bn_size)
            features.append(features_new)
        return tf.concat(features, -1)

    input_shape = (height, width, depth)

    inputs = Input(shape=input_shape)

    features = layer0(inputs)
    print("After first layer", features.shape)
    features = denseBlock(features, block_config[0], 32, 4)
    print("After first DensBlock", features.shape)
    num_features = num_init_features + growth_rate * block_config[0]

    ##trans layer
    features = trans_layer(features, num_features//2)
    print("After first Translayer", features.shape)

    ##second DenseBlock
    features = denseBlock(features, block_config[1], 32, 4)
    num_features = num_features//2 + growth_rate * block_config[1]
    print("After second DenseBlock", features.shape)
    ##trans layer
    features = trans_layer(features, num_features//2)
    print("After second Translayer", features.shape)

    ##third DenseBlock
    features = denseBlock(features, block_config[2], 32, 4)
    num_features = num_features//2 + growth_rate * block_config[2]
    print("After third DenseBlock", features.shape)
    ##trans layer
    features = trans_layer(features, num_features//2)
    print("After third Translayer", features.shape)

    ##forth DenseBlock
    features = denseBlock(features, block_config[3], 32, 4)
    num_features = num_features//2 + growth_rate * block_config[3]
    print("After forth DenseBlock", features.shape)


    ##final batch norm
    features = BatchNormalization(axis=chan_dim)(features)
    features = Activation("relu")(features)
    print("After relu", features.shape)
    #Global ave pooling
    outputs = GlobalAveragePooling2D()(features)
    print("After GlobalAveragePooling", outputs.shape)
    #outputs = Dense(4, activation='softmax')(outputs)
    outputs = Dense(1, activation='linear')(outputs)
    print("After dense", outputs.shape)

    model = Model(inputs, outputs, name="DenseNet")

    model.compile(optimizer=optimizers.Adam(1e-3),
                         loss=losses.mean_squared_error,
                         metrics=['mae'])
    return model 


def small_3d_model():
    inputs = tf.keras.layers.Input(shape=(176, 208, 176, 1), name='t1w_input')
    x = tf.keras.layers.Conv3D(filters=8, kernel_size=3, padding="valid", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling3D(pool_size=(3,3,3), strides=(3,3,3))(x)
    x = tf.keras.layers.Conv3D(filters=8, kernel_size=3, padding="valid", activation="relu")(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    print(x.shape)

    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv3D(filters=16, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)
    print(x.shape)
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    print(outputs.shape)

    model = Model(inputs, outputs, name="small_3d_model")

    model.compile(optimizer=optimizers.Adam(1e-3),
                         loss=losses.mean_squared_error,
                         metrics=['mae'])
    return model
