# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:52:44 2020

@author: benan
"""

import random as rn
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
#import sklearn.external.joblib as extjoblib
import joblib
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import (
    Dense,
    Input,
    Activation,
    Dropout,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Concatenate,
    BatchNormalization,
    concatenate,
)
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from utils.Lab_calculate import SingleValues
from utils.lambda_dataset import LambdaDataset
from utils.time import time_stamp
import cnn_bert_models
from tensorflow.keras.callbacks import LearningRateScheduler

def isNaN(num):
    return num != num

SEED = 123456
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
rn.seed(SEED)

def isNaN(num):
    return num != num

def divide_to_samples(df):
    sample_list = []
    for i in range(df.shape[0] // 445):
        sample_list.append(df.iloc[445 * i : 445 * i + 445])
    return sample_list

big_df_raw = pd.read_pickle("DATAFRAME/df_big_arranged.pkl")

big_df = big_df_raw

for column in range(1, 17):
    big_df.iloc[:, column] = big_df.iloc[:, column].str.split(" ").str[0]

# Delete the rows that doesnt include coating information
Ox_index = big_df.columns.get_loc("Ox")
nonna_values_we_need = big_df.shape[1] - Ox_index
thresh = (nonna_values_we_need,)
data_count_before = big_df.shape[0]
big_df = big_df.dropna(axis=0, thresh=thresh)
print("data_count_before: ", data_count_before)  # 329
data_count_after = big_df.shape[0]
print("data_count_after: ", data_count_after)  # 318

df = big_df
df = df.fillna(0)

def divide_sample(df):
    T_first = df.columns.get_loc("T_pre_0")
    Rc_first = df.columns.get_loc("Rc_pre_0")
    Ru_first = df.columns.get_loc("Ru_pre_0")
    T_last = df.columns.get_loc("T_pre_444")
    Rc_last = df.columns.get_loc("Rc_pre_444")
    Ru_last = df.columns.get_loc("Ru_pre_444")

    T = df.iloc[:, T_first : T_last + 1].values
    Rc = df.iloc[:, Rc_first : Rc_last + 1].values
    Ru = df.iloc[:, Ru_first : Ru_last + 1].values

    s = np.stack((T, Rc, Ru), axis=2)
    return s

train_valid_test_indexes = joblib.load("train_valid_test_indexes.pkl")
train_inx = train_valid_test_indexes["training_set_indexes"]
valid_inx = train_valid_test_indexes["valid_set_indexes"]
test_inx = train_valid_test_indexes["test_set_indexes"]

embedding_path = "BERT_embeddings/outputs/df_umap.pkl"
df_stack_embed = pd.read_pickle(embedding_path)

umap_embeddings =df_stack_embed.loc[:,["dim1","dim2"]]

save_result = True
estop_patience = 500
filters_1 = 128
kernel_size_1 = 15
epochs = 5000
lr_rate = 0.001
batch_size = 445
bert_style = embedding_path.split("/")[-1].split(".")[0]

cnn_hlayer = 2
bert_layer_size = 6
maxpooling = 4
maxp_strides = 4
bert_embedding_size = umap_embeddings.shape[1]
dense_layer_size = 512
dense_layer_size_2 = 256
model_name = "cnn_bert_model_5"  # "cnn_bert_model_6"

if "bert" in model_name:
    bert_usage = True
else:
    bert_usage = False

if "umap" in embedding_path:
    embedding_style = "umap"
elif "tsne" in embedding_path :
    embedding_style = "tsne"
elif "pca" in embedding_path :
    embedding_style = "pca"

df_bert_embeddings = umap_embeddings
df_bert_train = df_bert_embeddings.loc[train_inx, :].reset_index(drop=True)
df_bert_valid = df_bert_embeddings.loc[valid_inx, :].reset_index(drop=True)
df_bert_test = df_bert_embeddings.loc[test_inx, :].reset_index(drop=True)

sc_bert = StandardScaler()
X_train_bert = sc_bert.fit_transform(df_bert_train)
X_valid_bert = sc_bert.transform(df_bert_valid)
X_test_bert = sc_bert.transform(df_bert_test)

pre_first_inx = df.columns.get_loc("Visible_T_pre")
pre_last_inx = df.columns.get_loc("Solar_Ru_pre")  # UV_T_pre
post_first_inx = df.columns.get_loc("Visible_T_post")
post_last_inx = df.columns.get_loc("Solar_Ru_post")  # "UV_T_post"

df_train_raw, df_valid_raw, df_test_raw = (
    df.loc[train_inx, :],
    df.loc[valid_inx, :],
    df.loc[test_inx, :],
)

X_train_raw = divide_sample(df_train_raw)
y_train_raw = df_train_raw.iloc[:, post_first_inx : post_last_inx + 1]

X_valid_raw = divide_sample(df_valid_raw)
y_valid_raw = df_valid_raw.iloc[:, post_first_inx : post_last_inx + 1]

X_test_raw = divide_sample(df_test_raw)
y_test_raw = df_test_raw.iloc[:, post_first_inx : post_last_inx + 1]

X_train = np.zeros(X_train_raw.shape)
X_valid = np.zeros(X_valid_raw.shape)
X_test = np.zeros(X_test_raw.shape)

# Normalizing the input optical spectrum  and the output DOF values 
X_train = X_train_raw / 100
X_valid = X_valid_raw / 100
X_test = X_test_raw / 100

y_train = y_train_raw / 100
y_valid = y_valid_raw / 100
y_test = y_test_raw / 100

n_timesteps = 445
n_features = 3
n_outputs = 6

import cnn_models

try:
    del model
except:
    pass
if model_name == "cnn_bert_model_1":
    model = cnn_bert_models.cnn_bert_model_1(
        n_timesteps,
        n_features,
        filters_1,
        kernel_size_1,
        cnn_hlayer,
        bert_embedding_size,
        maxpooling,
        maxp_strides,
        bert_layer_size,
        n_outputs,
    )
    dense_layer_size = "*"
    dense_layer_size_2 = "*"
if model_name == "cnn_bert_model_2":
    model = cnn_bert_models.cnn_bert_model_2(
        n_timesteps,
        n_features,
        filters_1,
        kernel_size_1,
        cnn_hlayer,
        bert_embedding_size,
        maxpooling,
        maxp_strides,
        bert_layer_size,
        dense_layer_size,
        n_outputs,
    )
    dense_layer_size_2 = "*"

if model_name == "cnn_bert_model_3":
    model = cnn_bert_models.cnn_bert_model_3(
        n_timesteps,
        n_features,
        filters_1,
        kernel_size_1,
        cnn_hlayer,
        bert_embedding_size,
        maxpooling,
        maxp_strides,
        bert_layer_size,
        dense_layer_size,
        dense_layer_size_2,
        n_outputs,
    )

if model_name == "cnn_bert_model_4":
    model = cnn_bert_models.cnn_bert_model_4(
        n_timesteps,
        n_features,
        filters_1,
        kernel_size_1,
        cnn_hlayer,
        bert_embedding_size,
        maxpooling,
        maxp_strides,
        dense_layer_size,
        dense_layer_size_2,
        n_outputs,
    )
    bert_layer_size = "*"

if model_name == "cnn_bert_model_5":
    model = cnn_bert_models.cnn_bert_model_1(
        n_timesteps,
        n_features,
        filters_1,
        kernel_size_1,
        cnn_hlayer,
        bert_embedding_size,
        maxpooling,
        maxp_strides,
        bert_layer_size,
        n_outputs,
    )
    dense_layer_size = "*"
    dense_layer_size_2 = "*"

if model_name == "cnn_bert_model_6":
    model = cnn_bert_models.cnn_bert_model_1(
        n_timesteps,
        n_features,
        filters_1,
        kernel_size_1,
        cnn_hlayer,
        bert_embedding_size,
        maxpooling,
        maxp_strides,
        bert_layer_size,
        n_outputs,
    )
    dense_layer_size = "*"
    dense_layer_size_2 = "*"
if model_name == "cnn_model_1":
    model = cnn_models.cnn_model_1(
        n_timesteps, n_features, filters_1, kernel_size_1, cnn_hlayer
    )
    bert_layer_size = "*"
    maxpooling = "*"
    maxp_strides = "*"
    dense_layer_size = "*"
    dense_layer_size_2 = "*"
    bert_embedding_size = "*"

if model_name == "cnn_model_2":
    model = cnn_models.cnn_model_2(
        n_timesteps,
        n_features,
        filters_1,
        kernel_size_1,
        cnn_hlayer,
        maxpooling,
        maxp_strides,
        dense_layer_size,
    )
    bert_layer_size = "*"
    dense_layer_size_2 = "*"
    bert_embedding_size = "*"

def step_decay(epoch, lr):
    if epoch < 500:
        return lr
    else:
        if epoch % 50 == 0:
            print("lr: ", lr * 0.90)
            print("lr: ", lr * 0.90)
            print("lr: ", lr * 0.90)
            print("lr: ", lr * 0.90)
            print("lr: ", lr * 0.90)
            print("lr: ", lr * 0.90)
            return lr * 0.90
        else:
            return lr

early_stopping = EarlyStopping(
    monitor="val_loss", patience=estop_patience, restore_best_weights=True
)
lrate = LearningRateScheduler(step_decay)

adam = Adam(lr=lr_rate)
model.compile(optimizer=adam, loss="mae")
# fit model

def train_model():
    exp_time = time_stamp()
    if bert_usage:
        model.fit(
            [X_train, X_train_bert],
            y_train,
            validation_data=([X_valid, X_valid_bert], y_valid),
            epochs=epochs,
            verbose=1,
            batch_size=batch_size,
            callbacks=[early_stopping, lrate],
        )
        print("Bert embedding was used")

    else:
        model.fit(
            X_train,
            y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs,
            verbose=1,
            batch_size=batch_size,
            callbacks=[early_stopping],
        )
        print("Bert embedding was not used")
    return model, exp_time

def get_prediction(model, X_eva, y_eva, y_eva_raw):
    y_predict = model.predict(X_eva)
    y_predict2 = y_predict * 100
    y_pred_df = pd.DataFrame(y_predict2, columns=y_eva_raw.columns)
    post_single_values = y_eva_raw
    post_pred_single_values = y_pred_df

    return post_single_values, post_pred_single_values

def get_score(post_single_values, post_pred_single_values, df_eva_raw):
    post_single_values["id"] = df_eva_raw.index
    post_pred_single_values["id"] = df_eva_raw.index

    post_single_values["type"] = df_eva_raw["type"].values
    post_pred_single_values["type"] = df_eva_raw["type"].values

    post_single_values.index = pd.MultiIndex.from_arrays(
        post_single_values[["type", "id"]].values.T, names=["type", "id"]
    )
    post_pred_single_values.index = pd.MultiIndex.from_arrays(
        post_single_values[["type", "id"]].values.T, names=["type", "id"]
    )

    post_single_values.drop(["type", "id"], axis=1, inplace=True)
    post_pred_single_values.drop(["type", "id"], axis=1, inplace=True)

    df_mae = abs(post_pred_single_values - post_single_values)

    MAE = df_mae.mean()
    STD = df_mae.std()

    df_summary_score = pd.concat([MAE, STD], axis=1)
    df_summary_score = df_summary_score.rename(columns={0: "MAE", 1: "STD"})

    total_mae = sum(MAE)
    total_std = sum(STD)

    df_summary_score.loc["TOTAL", ["MAE"]] = total_mae
    df_summary_score.loc["TOTAL", ["STD"]] = total_std

    print("MAE: ", MAE)
    print("STD: ", STD)
    print("score: ", total_mae)
    return df_mae, df_summary_score

def train_and_save_all():
    cnn_model, exp_time = train_model()
    if bert_usage:
        post_dof_train, post_pred_dof_train = get_prediction(
            cnn_model, [X_train, X_train_bert], y_train, y_train_raw
        )
    else:
        post_dof_train, post_pred_dof_train = get_prediction(
            cnn_model, X_train, y_train, y_train_raw
        )

    df_mae_train, df_summary_score_train = get_score(
        post_dof_train, post_pred_dof_train, df_train_raw
    )

    if bert_usage:
        post_dof_valid, post_pred_dof_valid = get_prediction(
            cnn_model, [X_valid, X_valid_bert], y_valid, y_valid_raw
        )
    else:
        post_dof_valid, post_pred_dof_valid = get_prediction(
            cnn_model, X_valid, y_valid, y_valid_raw
        )

    df_mae_valid, df_summary_score_valid = get_score(
        post_dof_valid, post_pred_dof_valid, df_valid_raw
    )

    print("Experiment timestamp : ", exp_time)
    if save_result == True:
        joblib.dump(
            [post_dof_valid, post_pred_dof_valid],
            "experiment_results/pkl_files/post_pred_cnn_valid_{}.pkl".format(exp_time),
        )

        if bert_usage:
            post_dof_test, post_pred_dof_test = get_prediction(
                cnn_model, [X_test, X_test_bert], y_test_raw, y_test_raw
            )
        else:
            post_dof_test, post_pred_dof_test = get_prediction(
                cnn_model, X_test, y_test_raw, y_test_raw
            )

        df_mae_test, df_summary_score_test = get_score(
            post_dof_test, post_pred_dof_test, df_test_raw
        )
        joblib.dump(
            [post_dof_test, post_pred_dof_test],
            "experiment_results/pkl_files/post_pred_cnn_test_{}.pkl".format(exp_time),
        )

        with pd.ExcelWriter(
            "experiment_results/cnn/individual/cnn_model_{}.xlsx".format(exp_time)
        ) as writer:
            df_mae_valid.to_excel(writer, sheet_name="mean_absolute_error_valid")
            df_summary_score_valid.to_excel(writer, sheet_name="total_mae_valid")
            df_mae_test.to_excel(writer, sheet_name="mean_absolute_error_test")
            df_summary_score_test.to_excel(writer, sheet_name="total_mae_test")

    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=96,
    )

    df_exp = df_summary_score_valid[["MAE"]].T.reset_index(drop=True)

    df_exp["epoch"] = epochs
    df_exp["batch_size"] = batch_size
    df_exp["lr_rate"] = lr_rate
    df_exp["filters_1"] = filters_1
    df_exp["kernel_size_1"] = kernel_size_1

    df_exp["cnn_hlayer"] = cnn_hlayer
    df_exp["bert_layer_size"] = bert_layer_size
    df_exp["maxpooling"] = maxpooling
    df_exp["maxp_strides"] = maxp_strides
    df_exp["timestamp"] = exp_time
    df_exp["embedding_dim"] = bert_embedding_size
    df_exp["model_name"] = model_name
    df_exp["dense_layer_size"] = dense_layer_size
    df_exp["dense_layer_size_2"] = dense_layer_size_2
    df_exp["bert_usage"] = bert_usage
    df_exp["n_output"] = n_outputs

    if bert_usage:
        df_exp["bert_style"] = bert_style
    else:
        df_exp["bert_style"] = "*"

    if bert_usage:
        df_exp["embedding_style"] = bert_style
    else:
        df_exp["embedding_style"] = "*"

    if os.path.exists("experiment_results/cnn/cnn_experiments_all.csv"):
        df_exp.to_csv(
            "experiment_results/cnn/cnn_experiments_all.csv",
            mode="a",
            index=False,
            header=False,
        )
    else:
        df_exp.to_csv(
            "experiment_results/cnn/cnn_experiments_all.csv",
            mode="a",
            index=False,
            header=True,
        )

    result_dic = {
        "train_results": {
            "post_dof_train": post_dof_train,
            "post_pred_dof_train": post_pred_dof_train,
            "df_summary_score_train": df_summary_score_train,
        },
        "val_results": {
            "post_dof_valid": post_dof_valid,
            "post_pred_dof_valid": post_pred_dof_valid,
            "df_summary_score_valid": df_summary_score_valid,
        },
        "test_results": {
            "post_dof_test": post_dof_test,
            "post_pred_dof_test": post_pred_dof_test,
            "df_summary_score_test": df_summary_score_test,
        },
    }
    return result_dic

if __name__ == "__main__":
    cnn_model, exp_time = train_and_save_all()
