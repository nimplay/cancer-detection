import pandas as pd

cancer = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Cancer.csv')

"""
print(cancer.head())
print(cancer.info())
print(cancer.describe())
"""

# define target (y) and features (X)
cancer.columns
y= cancer['diagnosis']
x= cancer.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)

"""
# check for missing values
print(x.isnull().sum())
print(y.isnull().sum())
# check for duplicates
print(x.duplicated().sum())
print(y.duplicated().sum())
"""


# train test split
