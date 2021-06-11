#Załadowanie wszystkich potrzebnych bibliotek

from numpy.lib.financial import _irr_dispatcher
from pandas.core.algorithms import mode, value_counts
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy import stats

#Zaczytanie danych z pliku, dane pobrane z https://www.kaggle.com/mdjohirulislam/house-price-prediction-for-competition?fbclid=IwAR33qbuS3r85E2KHin8RAz0YJYvrd-PpGKqUmYTacl9BKjkGUP2-MRgUKUM
file_name_train = "train.csv"
file_name_test = "test.csv"
file_name_output = "house_prices_submission.csv"

train_data = pd.read_csv(file_name_train)
test_data = pd.read_csv(file_name_test)
id_data = test_data["Id"].to_frame()
df0 = train_data.copy()

#Wyświetlenie danych i ich podsumoawanie
dfdesc = train_data.describe().T
dfdesc


#Korelacja poszczególnych kolumn z ceną sprzedaży
train_data.corrwith(train_data["SalePrice"], method="pearson"

#Przedstawienie macierzy korelacji w postaci graficznej
X = train_data
test = test_data
y=X.SalePrice
X.head()

cor=X.corr()
high_cor=cor.index[cor["SalePrice"]>.5]
plt.figure(figsize=(9,9))
sns.heatmap(X[high_cor].corr(),annot=True,cmap='gnuplot2')


high_cor_feature = sns.PairGrid(X, y_vars=["SalePrice"], x_vars=["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF"], height=4)
high_cor_feature.map(sns.regplot)

low_cor_feature = sns.PairGrid(X, y_vars=["SalePrice"], x_vars=["LowQualFinSF","YrSold","OverallCond","MSSubClass","EnclosedPorch","KitchenAbvGr"], height=4)
low_cor_feature.map(sns.regplot)

train_data.info()

#Usuwamy kolumny w których jest duża ilość nulli, w większości przypadków miałyby jedynie negatywny wpływ na wynik

columns_to_drop = ["Id", "Alley", "FireplaceQu",
                   "PoolQC", "Fence", "MiscFeature"]
# Alley: 91 non-null
# FireplaceQu: 770
# PoolQC: 7
# Fence: 281
# MiscFeature: 54
df = train_data.drop(columns_to_drop, axis=1)
df2 = test_data.drop(columns_to_drop, axis=1)

#Przypisanie kolumn z typem 'object'

dfcategorics = df.select_dtypes(include=['object'])
print(dfcategorics.columns)
print()
print()

#Wypisanie ilości null w każdej z kolumn

dfnulls = df2.isnull().sum()
print(dfnulls)

#Wyświetlenie dla każdej kolumny wartości bez powtórzeń, oraz ich ilości

def display_value_counts(df1):
    column_names = """
    BsmtQual         44
    BsmtCond         45
    BsmtFinType1     42
    BsmtFinType2     42
    KitchenQual       1
    GarageFinish     78
    GarageQual       78
    GarageCond       78
    """.strip()

    for line in column_names.splitlines():
        column_name = line.split()[0]
        print()
        print(column_name)
        value_counts = df1[column_name].value_counts()
        print(value_counts)

display_value_counts(df)
display_value_counts(df2)

def fillna(df1):

    #Wypełnienie nullowych pól w kolumnach najczęstszymi wartościami
    df1["BsmtQual"] = df1["BsmtQual"].fillna("TA")
    df1["BsmtCond"] = df1["BsmtCond"].fillna("TA")
    df1["BsmtFinType1"] = df1["BsmtFinType1"].fillna("GLQ")
    df1["BsmtFinType2"] = df1["BsmtFinType2"].fillna("Unf")
    df1["KitchenQual"] = df1["KitchenQual"].fillna("TA")
    df1["GarageFinish"] = df1["GarageFinish"].fillna("Unf")
    df1["GarageQual"] = df1["GarageQual"].fillna("TA")
    df1["GarageCond"] = df1["GarageCond"].fillna("TA")

    #Wypełnienie nullowych pól w kolumnach interpolując nowe wartości
    df1["LotFrontage"] = df1["LotFrontage"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["MasVnrArea"] = df1["MasVnrArea"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["GarageYrBlt"] = df1["GarageYrBlt"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["BsmtFinSF1"] = df1["BsmtFinSF1"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["BsmtFinSF2"] = df1["BsmtFinSF2"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["BsmtUnfSF"] = df1["BsmtUnfSF"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["TotalBsmtSF"] = df1["TotalBsmtSF"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["BsmtFullBath"] = df1["BsmtFullBath"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["BsmtHalfBath"] = df1["BsmtHalfBath"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["GarageCars"] = df1["GarageCars"].interpolate(method='linear', direction = 'forward', inplace=True)
    df1["GarageArea"] = df1["GarageArea"].interpolate(method='linear', direction = 'forward', inplace=True)

fillna(df)
fillna(df2)

#normalizacja danych i wyświetlenie wykresu z danymi przed i po normalizacji

y=df.SalePrice
normalized_data=stats.boxcox(y)
fig,ax = plt.subplots(1,2)
sns.distplot(y,ax=ax[0])
ax[0].set_title("orginal data")
sns.distplot(normalized_data[0] , ax=ax[1])
ax[1].set_title("normalized data")


#Wybranie wartości typu int, float i konkretnych kolumn. Później sklejenie całości.

def build_df(df1):
    df = df1.copy()
    dfi = df.select_dtypes(include=['int64'])
    dff = df.select_dtypes(include=['float64'])

    ordinal_columns = """
ExterQual
ExterCond
BsmtQual
BsmtCond
BsmtFinType1
BsmtFinType2
HeatingQC
KitchenQual
GarageFinish
GarageQual
GarageCond
PavedDrive
    """.strip().split("\n")

    dfo = df[ordinal_columns]
    dfx = pd.concat([dfi, dfo, dff], axis=1)
    return dfx

df = build_df(df)
df2 = build_df(df2)


# Zamiana wartości tekstowych na liczbowe.

def encode_ordinal(df0, column_name: str, categories):
    df = df0.copy()
    mapping = {}
    for i, category in enumerate(categories):
        mapping[category] = i

    df[column_name] = df[column_name].map(mapping)
    return df


def encode_ordinal_columns(df0):
    df1 = df0.copy()
    values_qualities = ["Ex", "Gd", "TA", "Fa", "Po"]
    values_qualities_with_na = ["NA", "Ex", "Gd", "TA", "Fa", "Po"]
    values_fintype = ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
    values_garagefinish = ["NA", "Unf", "RFn", "Fin"]
    values_paveddrive = ["N", "P", "Y"]

    df1 = encode_ordinal(df1, "ExterQual", values_qualities)
    df1 = encode_ordinal(df1, "ExterCond", values_qualities)
    df1 = encode_ordinal(df1, "BsmtQual", values_qualities_with_na)
    df1 = encode_ordinal(df1, "BsmtCond", values_qualities_with_na)
    df1 = encode_ordinal(df1, "BsmtFinType1", values_fintype)
    df1 = encode_ordinal(df1, "BsmtFinType2", values_fintype)
    df1 = encode_ordinal(df1, "HeatingQC", values_qualities)
    df1 = encode_ordinal(df1, "KitchenQual", values_qualities)
    df1 = encode_ordinal(df1, "GarageFinish", values_garagefinish)
    df1 = encode_ordinal(df1, "GarageQual", values_qualities_with_na)
    df1 = encode_ordinal(df1, "GarageCond", values_qualities_with_na)
    df1 = encode_ordinal(df1, "PavedDrive", values_paveddrive)

    return df1


df = encode_ordinal_columns(df)
df2 = encode_ordinal_columns(df2)

#Sprawdzenie wartości nullowych

dfnulls1 = pd.isnull(df).sum()
dfnulls2 = pd.isnull(df2).sum()
print('dfnulls1:')
print(dfnulls1)
print()
print('dfnulls2:')
print(dfnulls2)
#%% train test split

target = "SalePrice"
x = df.drop(target, axis=1)
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=18)

#ustawienie kilku różnych modeli, następnie wytrenowanie modeli na podstawie treningowych danych.

models = [LinearRegression(), RandomForestRegressor(random_state=23),
          GradientBoostingRegressor(n_estimators=500, random_state=23)]

for model in models:
    model.fit(x_train, y_train)
    print("_" * 20)
    print(str(model))
    score = model.score(x_test, y_test)
    print(score)

#Najlepszy wynik miał GradientBoostingRegressor, więc jego używamy to wyliczenia danych i zapisanie do pliku

model = GradientBoostingRegressor(n_estimators=500, random_state=23)
model.fit(x_train, y_train)

predictions = model.predict(df2)

my_dict = {"Id": id_data["Id"], "SalePrice": predictions}
dfout = pd.DataFrame(my_dict)
dfout.to_csv(file_name_output, index=False)