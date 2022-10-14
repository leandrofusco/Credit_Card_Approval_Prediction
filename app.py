from help_functions import calc_day_of_birth, calc_day_of_employed, get_appartment, calculate_age, get_label_for_data
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# Pre Processing
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd
import numpy as np

def data_cleaning(df):
    df.drop_duplicates(inplace=True)
    # filling null data 
    df['OCCUPATION_TYPE'] =df['OCCUPATION_TYPE'].replace(np.nan,'others')
    return df

def feature_engineering(df):
    #gerar a data de aniversário de cada pessoa 
    df['BIRTH_DAY']   = df['DAYS_BIRTH'].apply(calc_day_of_birth)
    # calcular o dia que foi contratado  
    df['EMPLOYED_DAY']   = df['DAYS_EMPLOYED'].apply(calc_day_of_employed)
    #Melhorar nomes de house/apartment
    df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].apply(get_appartment)
    #calcular idade
    df['age'] = df['BIRTH_DAY'].apply(calculate_age)
    #melhorar nome do grau educacional
    df['NAME_EDUCATION_TYPE'] =df['NAME_EDUCATION_TYPE'].apply(get_ducational_type)

    # substituindo os valores C,X por números 
    df['STATUS'].replace({'C': 6, 'X' : 7}, inplace=True)
    df['STATUS']=df['STATUS'].astype(int)

    # criando a coluna TARGET
    df['TARGET'] = df['STATUS'].apply(get_label_for_data)

    return df

def drop_columns(df):
    #excluindo colunas desnessessárias 
    df = df.drop(['ID','DAYS_BIRTH','MONTHS_BALANCE','FLAG_WORK_PHONE','DAYS_EMPLOYED','EMPLOYED_DAY','BIRTH_DAY'],axis=1)
    
    return df

def preprocessing(df):
    column_data = ["TARGET","CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]
    for col in column_data:
        label = LabelEncoder()
        df[col] = label.fit_transform(df[col].values)
    df = pd.get_dummies(df, drop_first=True, columns=['OCCUPATION_TYPE'])

    return df

def preparation_spliting(df):
    x = df.drop(['TARGET'], axis=True)
    y = df['TARGET']
    # data standarization 
    sc=StandardScaler()
    x_scaled = sc.fit_transform(x)

    return x_scaled

def dimensionality_reduction_by_pca(x_scaled):
    pca = PCA()
    pct = pca.fit_transform(x_scaled)


def data_balancing(x_scaled,y):
    undersample = RandomUnderSampler(random_state=0)
    X, y = undersample.fit_resample(x_scaled, y)


def predict(df):
