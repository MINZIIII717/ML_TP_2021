import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler

# ----------------------------------- Functions -----------------------------------
def nativeCountry(x) :
  if x != "United-States" :
    return str(x).replace(str(x),"0")
  return "1"

def income(x) :
  if x == '<=50K' :
    return x.replace(x,"0")
  return "1"


# Load the dataset (adult.csv)
df = pd.read_csv('adult.csv')

# ----------------------------------- Preprocessing -----------------------------------
# 1. Change the ? value to NaN
df = df.replace('?', np.NaN)

# 2. Drop the NaN values (row)
df = df.dropna(axis=0)

# 3. Drop columns (education, capital-gain, capital-loss)
df.drop(['education', 'capital-gain', 'capital-loss'], axis=1, inplace=True)

# 4. Change "native-contry" values to binary value
# "United-States" : 1, not "United-States" : 0
df["native-country"]=df["native-country"].apply(nativeCountry)

# 5. Change "income" values to binary value
# <=50k : 2, >50k :1
df["income"]=df["income"].apply(income)

# 6. Change educational number to three sector
# <10 : 1, 10~13 : 2, >13 :3
df["educational-num"]=df["educational-num"].mask(df["educational-num"] < 10, 1)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] == 10, 2)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] == 11, 2)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] == 12, 2)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] == 13, 2)
df["educational-num"]=df["educational-num"].mask(df["educational-num"] > 13, 3)

pd.set_option('display.max_columns', None)
print(df)
