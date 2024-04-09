# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:25:06 2022

@author: benan
"""

import sys

from xlrd import open_workbook
import matplotlib.pyplot as plt
import random as rn
import numpy as np
import statistics
import keras
import tensorflow
import joblib
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Dense,Input,Activation,LeakyReLU
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import copy
import xlwt
from matplotlib.ticker import NullFormatter
from sklearn.model_selection import KFold
import tensorflow as tf
import glob
import pandas as pd
import pickle
import seaborn as sns
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Dense,Input,Activation,Dropout,Conv1D,MaxPooling1D,Flatten,Concatenate,BatchNormalization,concatenate
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from utils.Lab_calculate import SingleValues
from utils.lambda_dataset import LambdaDataset
from utils.time import time_stamp

def cnn_bert_model_1(n_timesteps,n_features,filters_1,kernel_size_1,cnn_hlayer,bert_embedding_size,maxpooling,maxp_strides,bert_layer_size,n_outputs):
    opt_input = Input(shape=(n_timesteps,n_features))  ## branch 1 with image input
    x = Conv1D(filters_1, kernel_size_1, padding='same')(opt_input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = LeakyReLU(alpha=0.1)(x)
    for i in range(cnn_hlayer):
        x = MaxPooling1D(pool_size = maxpooling, strides = maxp_strides)(x)
        x = Conv1D(filters_1, kernel_size_1, padding='same')(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    
        x = LeakyReLU(alpha=0.1)(x)
    out_1 = Flatten()(x)
    
    bert_input = Input(shape=(bert_embedding_size,))  
    
    out_2 = Dense(bert_layer_size, activation='relu')(bert_input)
    
    concatenated = concatenate([out_1, out_2])
    
    
    output = Dense(n_outputs, activation='linear')(concatenated)
    

    
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True,
    #     initial_value_threshold = 0.06,
    #     verbose = 1)
    model = Model([opt_input, bert_input], output)
    return model


def cnn_bert_model_2(n_timesteps,n_features,filters_1,kernel_size_1,cnn_hlayer,bert_embedding_size,maxpooling,maxp_strides,bert_layer_size,dense_layer_size,n_outputs):
    opt_input = Input(shape=(n_timesteps,n_features))  ## branch 1 with image input
    x = Conv1D(filters_1, kernel_size_1, padding='same')(opt_input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = LeakyReLU(alpha=0.1)(x)
    for i in range(cnn_hlayer):
        x = MaxPooling1D(pool_size = maxpooling, strides = maxp_strides)(x)
        x = Conv1D(filters_1, kernel_size_1, padding='same')(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    
        x = LeakyReLU(alpha=0.1)(x)
    out_1 = Flatten()(x)
    out_1 = Dense(dense_layer_size, activation='relu')(out_1)
    
    
    bert_input = Input(shape=(bert_embedding_size,))  
    
    out_2 = Dense(bert_layer_size, activation='relu')(bert_input)    
    concatenated = concatenate([out_1, out_2])    
    
    output = Dense(n_outputs, activation='linear')(concatenated)

    model = Model([opt_input, bert_input], output)
    return model

def cnn_bert_model_3(n_timesteps,n_features,filters_1,kernel_size_1,cnn_hlayer,bert_embedding_size,maxpooling,maxp_strides,bert_layer_size,dense_layer_size,dense_layer_size_2,n_outputs):
    
    opt_input = Input(shape=(n_timesteps,n_features))  ## branch 1 with image input
    x = Conv1D(filters_1, kernel_size_1, padding='same')(opt_input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = LeakyReLU(alpha=0.1)(x)
    for i in range(cnn_hlayer):
        x = MaxPooling1D(pool_size = maxpooling, strides = maxp_strides)(x)
        x = Conv1D(filters_1, kernel_size_1, padding='same')(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    
        x = LeakyReLU(alpha=0.1)(x)
    out_1 = Flatten()(x)
    out_1 = Dense(dense_layer_size, activation='relu')(out_1)
    out_1 = Dense(dense_layer_size_2, activation='relu')(out_1)
    
    bert_input = Input(shape=(bert_embedding_size,))  
    
    out_2 = Dense(bert_layer_size, activation='relu')(bert_input)    
    concatenated = concatenate([out_1, out_2])    
    
    output = Dense(n_outputs, activation='linear')(concatenated)

    model = Model([opt_input, bert_input], output)
    
    
    
    return model


def cnn_bert_model_4(n_timesteps,n_features,filters_1,kernel_size_1,cnn_hlayer,bert_embedding_size,maxpooling,maxp_strides,dense_layer_size,dense_layer_size_2,n_outputs):
    opt_input = Input(shape=(n_timesteps,n_features))  ## branch 1 with image input
    x = Conv1D(filters_1, kernel_size_1, padding='same')(opt_input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = LeakyReLU(alpha=0.1)(x)
    for i in range(cnn_hlayer):
        x = MaxPooling1D(pool_size = maxpooling, strides = maxp_strides)(x)
        x = Conv1D(filters_1, kernel_size_1, padding='same')(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    
        x = LeakyReLU(alpha=0.1)(x)
    out_1 = Flatten()(x)
    out_1 = Dense(dense_layer_size, activation='relu')(out_1)
    
    
    bert_input = Input(shape=(bert_embedding_size,))  
    
    out_2 = bert_input
    concatenated = concatenate([out_1, out_2])    
    
    output = Dense(n_outputs, activation='linear')(concatenated)

    model = Model([opt_input, bert_input], output)
    return model

def cnn_bert_model_5(n_timesteps,n_features,filters_1,kernel_size_1,cnn_hlayer,bert_embedding_size,maxpooling,maxp_strides,bert_layer_size,n_outputs):
    opt_input = Input(shape=(n_timesteps,n_features))  ## branch 1 with image input
    x = Conv1D(filters_1, kernel_size_1, padding='same')(opt_input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = LeakyReLU(alpha=0.1)(x)
    for i in range(cnn_hlayer):
        x = MaxPooling1D(pool_size = maxpooling, strides = maxp_strides)(x)
        x = tf.keras.layers.Dropout(.3,)(x)
        x = Conv1D(filters_1, kernel_size_1, padding='same')(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
        
        x = LeakyReLU(alpha=0.1)(x)
    
    out_1 = Flatten()(x)
    
    bert_input = Input(shape=(bert_embedding_size,))  
    
    out_2 = Dense(2, activation='linear')(bert_input)
    
    concatenated = concatenate([out_1, out_2])
    concatenated2 = Dense(n_outputs, activation='relu')(concatenated)
    
    output = Dense(n_outputs, activation='linear')(concatenated2)
    

    
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True,
    #     initial_value_threshold = 0.06,
    #     verbose = 1)
    model = Model([opt_input, bert_input], output)
    return model


def cnn_bert_model_6(n_timesteps,n_features,filters_1,kernel_size_1,cnn_hlayer,bert_embedding_size,maxpooling,maxp_strides,bert_layer_size,n_outputs):
    opt_input = Input(shape=(n_timesteps,n_features))  ## branch 1 with image input
    x = Conv1D(filters_1, kernel_size_1, padding='same')(opt_input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = LeakyReLU(alpha=0.1)(x)
    for i in range(cnn_hlayer):
        x = MaxPooling1D(pool_size = maxpooling, strides = maxp_strides)(x)
        x = tf.keras.layers.Dropout(.3,)(x)
        x = Conv1D(filters_1, kernel_size_1, padding='same')(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
        
        x = LeakyReLU(alpha=0.1)(x)
    
    out_1 = Flatten()(x)
    
    bert_input = Input(shape=(bert_embedding_size,))  
    
    out_2 = Dense(2, activation='linear')(bert_input)
    
    concatenated = concatenate([out_1, bert_input])
    concatenated2 = Dense(2*n_outputs, activation='relu')(concatenated)
    
    output = Dense(n_outputs, activation='linear')(concatenated2)
    

    
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True,
    #     initial_value_threshold = 0.06,
    #     verbose = 1)
    model = Model([opt_input, bert_input], output)
    return model



def cnn_bert_model_7(n_timesteps,n_features,filters_1,kernel_size_1,cnn_hlayer,bert_embedding_size,maxpooling,maxp_strides,bert_layer_size,n_outputs):
    opt_input = Input(shape=(n_timesteps,n_features))  ## branch 1 with image input
    x = Conv1D(filters_1, kernel_size_1, padding='same')(opt_input)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
    x = LeakyReLU(alpha=0.1)(x)
    for i in range(cnn_hlayer):
        x = MaxPooling1D(pool_size = maxpooling, strides = maxp_strides)(x)
        x = tf.keras.layers.Dropout(.3,)(x)
        x = Conv1D(filters_1, kernel_size_1, padding='same')(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform")(x)
        
        x = LeakyReLU(alpha=0.1)(x)
    
    out_1 = Flatten()(x)
    
    bert_input = Input(shape=(bert_embedding_size,))  
    
    out_2 = Dense(2, activation='linear')(bert_input)
    
    concatenated = concatenate([out_1, bert_input])
    concatenated2 = Dense(2*n_outputs, activation='relu')(concatenated)
    
    output = Dense(n_outputs, activation='linear')(concatenated2)
    

    
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True,
    #     initial_value_threshold = 0.06,
    #     verbose = 1)
    model = Model([opt_input, bert_input], output)
    return model