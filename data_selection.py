# %%
import numpy as np
import pandas as pd
import random as rn
import joblib


def isNaN(num):
    return num != num


import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

pd.options.display.max_colwidth = 2000
pd.options.display.max_rows = 2000

pd.options.display.max_columns = 2000
# %%

# pre_spectrum_dataset = pd.read_pickle("C:\\Users\\benan\\OneDrive\\12_SISECAM\DATAFRAME\\pre_spectrum_dataset_big_arranged.pkl")#%% Thickness

import joblib

pre_spectrum_dataset = joblib.load("pre_spectrum_values.h5")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Prepare empty lists to collect train, test, and validation indices
train_indices = []
test_indices = []
val_indices = []

# Loop through each group and split the data
for group in pre_spectrum_dataset["Coating Type"].unique():
    # Filter data for the current group
    group_data = pre_spectrum_dataset[pre_spectrum_dataset["Coating Type"] == group]

    # First split: 80% train, 20% for temp (test + validation)
    train_data, temp_data = train_test_split(group_data, test_size=0.2, random_state=42)

    # Second split on temp_data: 50% test, 50% validation
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Collect indices
    train_indices.extend(train_data.index)
    test_indices.extend(test_data.index)
    val_indices.extend(val_data.index)

# Create data subsets
train_set = pre_spectrum_dataset.loc[train_indices]
test_set = pre_spectrum_dataset.loc[test_indices]
validation_set = pre_spectrum_dataset.loc[val_indices]

# Display the sizes to verify correctness
print(f"Training Set: {len(train_set)} rows")
print(f"Test Set: {len(test_set)} rows")
print(f"Validation Set: {len(validation_set)} rows")

# Optional: Check group distribution in each set
print("Group distribution in Training Set:")
print(train_set["Coating Type"].value_counts(normalize=True))
print("Group distribution in Test Set:")
print(test_set["Coating Type"].value_counts(normalize=True))
print("Group distribution in Validation Set:")
print(validation_set["Coating Type"].value_counts(normalize=True))


train_valid_test_indexes = {
    "training_set_indexes": train_indices,
    "valid_set_indexes": test_indices,
    "test_set_indexes": val_indices,
}

joblib.dump(train_valid_test_indexes, "train_valid_test_indexes.pkl")
# %%
