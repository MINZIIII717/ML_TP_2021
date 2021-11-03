import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler


# Load the dataset (adult.csv)
df = pd.read_csv('adult.csv')

# ----------------------------------- Preprocessing -----------------------------------
# 1. Change the ? value to NaN
df = df.replace('?', np.NaN)

# 2. Drop the NaN values (row)
df = df.dropna(axis=0)