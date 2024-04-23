# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:52:44 2020

@author: benan
"""
# %%
import os
import joblib
import numpy as np
import pandas as pd
import random as rn

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
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
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler

from utils.Lab_calculate import SingleValues
from utils.lambda_dataset import LambdaDataset
from utils.time import time_stamp
import cnn_bert_models

# Configure Pandas display options for readability
pd.options.display.max_columns = 2000
pd.options.display.max_rows = 2000
pd.options.display.max_colwidth = 2000

# Set a consistent seed for reproducibility
SEED = 123456
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
rn.seed(SEED)
# Load and prepare data
file_paths = {
    "pre_spectrum": "DATA_BUCKET/pre_spectrum_values.h5",
    "post_single": "DATA_BUCKET/post_single_values.h5",
    "indexes": "DATA_BUCKET/train_valid_test_indexes.pkl",
    "embedding": "DATA_BUCKET/umap_embedding.h5",
}
embedding_path = file_paths["embedding"]


def extract_features(dataframe, prefix_list, index_range):
    """
    Extracts columns from a DataFrame based on prefix patterns and an index range.
    Columns for each feature are combined into a single three-dimensional array.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame from which to extract data.
        prefix_list (list of str): List of prefixes to extract (e.g., ['T_pre_', 'Rc_pre_', 'Ru_pre_']).
        index_range (tuple): Start and end index for extraction.

    Returns:
        np.array: A stacked numpy array of the extracted features across the specified range.
    """
    features = []
    for prefix in prefix_list:
        start_col = dataframe.columns.get_loc(f"{prefix}{index_range[0]}")
        end_col = dataframe.columns.get_loc(f"{prefix}{index_range[1]}") + 1
        feature_data = dataframe.iloc[:, start_col:end_col].values
        features.append(feature_data)
    return np.stack(features, axis=2)


def load_and_prepare_data(file_paths):
    """
    Loads datasets and prepares them for training by extracting and normalizing features,
    and processes UMAP embeddings similar to BERT embeddings for use in neural network models.

    Parameters:
        file_paths (dict): Dictionary containing paths to the necessary files.

    Returns:
        dict: Dictionary containing prepared training, validation, and test sets, including BERT-like embeddings.
    """
    pre_spectrum_dataset = joblib.load(file_paths["pre_spectrum"])
    post_single_dataset = joblib.load(file_paths["post_single"])
    train_valid_test_indexes = joblib.load(file_paths["indexes"])
    df_stack_embed = joblib.load(file_paths["embedding"])

    train_idx = train_valid_test_indexes["training_set_indexes"]
    valid_idx = train_valid_test_indexes["valid_set_indexes"]
    test_idx = train_valid_test_indexes["test_set_indexes"]

    df_train_x = pre_spectrum_dataset.loc[train_idx]
    df_valid_x = pre_spectrum_dataset.loc[valid_idx]
    df_test_x = pre_spectrum_dataset.loc[test_idx]

    X_train_raw = extract_features(
        df_train_x, ["T_pre_", "Rc_pre_", "Ru_pre_"], (0, 444)
    )
    X_valid_raw = extract_features(
        df_valid_x, ["T_pre_", "Rc_pre_", "Ru_pre_"], (0, 444)
    )
    X_test_raw = extract_features(df_test_x, ["T_pre_", "Rc_pre_", "Ru_pre_"], (0, 444))

    X_train = X_train_raw / 100
    X_valid = X_valid_raw / 100
    X_test = X_test_raw / 100

    y_train = post_single_dataset.loc[train_idx].iloc[:, 1:] / 100
    y_valid = post_single_dataset.loc[valid_idx].iloc[:, 1:] / 100
    y_test = post_single_dataset.loc[test_idx].iloc[:, 1:] / 100

    df_bert_embeddings = df_stack_embed.loc[:, ["dim1", "dim2"]]
    sc = StandardScaler()
    X_train_bert = sc.fit_transform(
        df_bert_embeddings.loc[train_idx].reset_index(drop=True)
    )
    X_valid_bert = sc.transform(
        df_bert_embeddings.loc[valid_idx].reset_index(drop=True)
    )
    X_test_bert = sc.transform(df_bert_embeddings.loc[test_idx].reset_index(drop=True))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_train_bert": X_train_bert,
        "X_valid": X_valid,
        "X_valid_bert": X_valid_bert,
        "y_valid": y_valid,
        "X_test": X_test,
        "X_test_bert": X_test_bert,
        "y_test": y_test,
        "df_train_x": df_train_x,
        "df_valid_x": df_valid_x,
        "df_test_x": df_test_x,
    }


def create_model_config():
    """
    Configures and returns the settings for a CNN-BERT model as a dictionary.

    Returns:
        dict: A dictionary containing key-value pairs of model configuration parameters,
        including the number of timesteps, features, outputs, layer specifics, and training control parameters.
    """
    config = {
        "n_timesteps": 445,
        "n_features": 3,
        "n_outputs": 6,
        "save_result": True,
        "estop_patience": 30,
        "filters_1": 128,
        "kernel_size_1": 15,
        "epochs": 10,
        "lr_rate": 0.001,
        "batch_size": 445,
        "bert_style": embedding_path.split("/")[-1].split(".")[0],
        "cnn_hlayer": 2,
        "bert_layer_size": 6,
        "maxpooling": 4,
        "maxp_strides": 4,
        "bert_embedding_size": 2,
        "dense_layer_size": 512,
        "dense_layer_size_2": 256,
        "model_name": "cnn_bert_model_5",
    }
    return config


def create_model(config):
    """
    Constructs a CNN-BERT hybrid model using the specifications provided in the config dictionary.

    Parameters:
        config (dict): A dictionary containing the configuration settings for the model.

    Returns:
        TensorFlow/Keras model: A compiled model ready for training.
    """
    model = cnn_bert_models.cnn_bert_model_1(
        config["n_timesteps"],
        config["n_features"],
        config["filters_1"],
        config["kernel_size_1"],
        config["cnn_hlayer"],
        config["bert_embedding_size"],
        config["maxpooling"],
        config["maxp_strides"],
        config["bert_layer_size"],
        config["n_outputs"],
    )
    adam_optimizer = Adam(learning_rate=config["lr_rate"])
    model.compile(optimizer=adam_optimizer, loss="mae")
    return model


def step_decay(epoch, lr):
    """
    Learning rate scheduler that reduces the learning rate by 10% every 50 epochs after the first 500 epochs.

    Parameters:
        epoch (int): Current epoch during training.
        lr (float): Current learning rate.

    Returns:
        float: Adjusted learning rate.
    """
    if epoch >= 500 and epoch % 50 == 0:
        new_lr = lr * 0.90
        print(f"Reducing learning rate to: {new_lr:.6f}")
        return new_lr
    return lr


def train_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    config,
    bert_usage,
    X_train_bert=None,
    X_valid_bert=None,
):
    """
    Trains the neural network model with optional BERT embeddings and implements early stopping and dynamic learning rate adjustments.

    Parameters:
        model: The neural network model to be trained.
        X_train, y_train: Arrays containing the training data and labels.
        X_valid, y_valid: Arrays containing the validation data and labels.
        config (dict): Configuration dictionary with training parameters.
        bert_usage (bool): Flag indicating whether BERT embeddings are used.
        X_train_bert, X_valid_bert (optional): BERT embeddings for training and validation.

    Returns:
        model: The trained model.
        exp_time: Timestamp at which the training was executed.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=config["estop_patience"], restore_best_weights=True
    )
    learning_rate_scheduler = LearningRateScheduler(step_decay)
    training_start_time = time_stamp()

    training_inputs = [X_train, X_train_bert] if bert_usage else X_train
    validation_inputs = [X_valid, X_valid_bert] if bert_usage else X_valid

    model.fit(
        training_inputs,
        y_train,
        validation_data=(validation_inputs, y_valid),
        epochs=config["epochs"],
        verbose=1,
        batch_size=config["batch_size"],
        callbacks=[early_stopping, learning_rate_scheduler],
    )
    return model, training_start_time


def get_prediction(model, X_evaluation, y_true):
    """
    Generates predictions for the evaluation dataset and formats them alongside the true labels.

    Parameters:
        model: The trained model.
        X_evaluation: The feature data for prediction.
        y_true: The true labels for formatting.

    Returns:
        tuple: A tuple containing the true labels and a DataFrame of the predictions aligned with the true labels.
    """
    predictions = model.predict(X_evaluation)
    predictions_scaled = predictions * 100
    predictions_dataframe = pd.DataFrame(predictions_scaled, columns=y_true.columns)
    return y_true, predictions_dataframe


def get_score(true_values, predicted_values, evaluation_data_raw):
    """
    Computes the Mean Absolute Error (MAE) and Standard Deviation (STD) between predicted and true values.

    Parameters:
        true_values: DataFrame containing the true output values.
        predicted_values: DataFrame containing the predicted output values.
        evaluation_data_raw: Raw evaluation data used for additional indexing or information.

    Returns:
        tuple: A tuple containing a DataFrame of MAEs per feature and a summary DataFrame with MAEs and STDs.
    """
    true_values["id"] = evaluation_data_raw.index
    predicted_values["id"] = evaluation_data_raw.index
    true_values["Coating Type"] = evaluation_data_raw["Coating Type"].values
    predicted_values["Coating Type"] = evaluation_data_raw["Coating Type"].values

    multi_index_columns = ["Coating Type", "id"]
    true_values.set_index(multi_index_columns, inplace=True)
    predicted_values.set_index(multi_index_columns, inplace=True)

    mae_dataframe = abs(predicted_values - true_values)
    mean_absolute_error = mae_dataframe.mean()
    standard_deviation = mae_dataframe.std()
    summary_score_dataframe = pd.concat(
        [mean_absolute_error, standard_deviation], axis=1
    ).rename(columns={0: "MAE", 1: "STD"})

    summary_score_dataframe.loc["TOTAL", ["MAE", "STD"]] = [
        mean_absolute_error.sum(),
        standard_deviation.sum(),
    ]

    print("MAE: ", mean_absolute_error)
    print("STD: ", standard_deviation)
    return mae_dataframe, summary_score_dataframe


def train_and_save_all(save_result=True):
    """
    Manages the complete training process, evaluates model performance, and optionally saves results.
    This function sets up the environment, configures the model, performs training, and evaluates it on
    both validation and test datasets. Results can be saved to files for further analysis.

    Parameters:
        save_result (bool): If True, saves the training, validation, and test predictions and performance metrics to files.

    Returns:
        dict: A dictionary containing detailed training, validation, and test results including the model's predictions and performance scores.
    """
    # Load configurations and initialize the model
    config = create_model_config()
    model = create_model(config)
    bert_usage = "bert" in config["model_name"]
    data = load_and_prepare_data(file_paths)
    # Data preparation
    X_train, y_train = data["X_train"], data["y_train"]
    X_train_bert, X_valid_bert, X_test_bert = (
        data["X_train_bert"],
        data["X_valid_bert"],
        data["X_test_bert"],
    )
    X_valid, y_valid = data["X_valid"], data["y_valid"]
    X_test, y_test = data["X_test"], data["y_test"]
    df_train_x, df_valid_x, df_test_x = (
        data["df_train_x"],
        data["df_valid_x"],
        data["df_test_x"],
    )

    # Model training
    cnn_model, experiment_time = train_model(
        model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        config,
        bert_usage,
        X_train_bert,
        X_valid_bert,
    )

    # Model prediction and evaluation for training set
    training_inputs = [X_train, X_train_bert] if bert_usage else X_train
    post_dof_train, post_pred_dof_train = get_prediction(
        cnn_model, training_inputs, y_train
    )
    df_mae_train, df_summary_score_train = get_score(
        post_dof_train, post_pred_dof_train, df_train_x
    )

    # Model prediction and evaluation for validation set
    validation_inputs = [X_valid, X_valid_bert] if bert_usage else X_valid
    post_dof_valid, post_pred_dof_valid = get_prediction(
        cnn_model, validation_inputs, y_valid
    )
    df_mae_valid, df_summary_score_valid = get_score(
        post_dof_valid, post_pred_dof_valid, df_valid_x
    )

    # Logging experiment time
    print("Experiment timestamp: ", experiment_time)

    # Saving results if enabled
    if save_result:
        joblib.dump(
            [post_dof_valid, post_pred_dof_valid],
            f"experiment_results/pkl_files/post_pred_cnn_valid_{experiment_time}.pkl",
        )

        # Prediction and evaluation for test set
        test_inputs = [X_test, X_test_bert] if bert_usage else X_test
        post_dof_test, post_pred_dof_test = get_prediction(
            cnn_model, test_inputs, y_test
        )
        df_mae_test, df_summary_score_test = get_score(
            post_dof_test, post_pred_dof_test, df_test_x
        )
        joblib.dump(
            [post_dof_test, post_pred_dof_test],
            f"experiment_results/pkl_files/post_pred_cnn_test_{experiment_time}.pkl",
        )

        with pd.ExcelWriter(
            f"experiment_results/cnn/individual/cnn_model_{experiment_time}.xlsx"
        ) as writer:
            df_mae_valid.to_excel(writer, sheet_name="mean_absolute_error_valid")
            df_summary_score_valid.to_excel(writer, sheet_name="total_mae_valid")
            df_mae_test.to_excel(writer, sheet_name="mean_absolute_error_test")
            df_summary_score_test.to_excel(writer, sheet_name="total_mae_test")

    # Creating experiment summary
    df_exp = df_summary_score_valid[["MAE"]].T.reset_index(drop=True)
    for key, value in config.items():
        df_exp[key] = value
    df_exp["timestamp"] = experiment_time
    df_exp["bert_usage"] = bert_usage

    # Saving experiment summary to CSV
    exp_results_file = "experiment_results/cnn/cnn_experiments_all.csv"
    if os.path.exists(exp_results_file):
        df_exp.to_csv(exp_results_file, mode="a", index=False, header=False)
    else:
        df_exp.to_csv(exp_results_file, mode="a", index=False, header=True)

    # Compile all results into a dictionary for possible further processing
    result_dict = {
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
    return result_dict


if __name__ == "__main__":
    results_dictionary = train_and_save_all()
    print("Experiment completed successfully.")

# %%
