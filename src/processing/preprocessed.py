import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
def preprocess_data(data_path, model_save_dir):
    # Load the dataset
    initial_dataset = pd.read_csv(data_path)
    initial_dataset.drop_duplicates(inplace=True)
#Plan for EDA
#0. We are not extracting date,month from timestamp as this is a synthetically generated data and the timestamps are pretty close to each other (basically the same timestamp)
#1. Remove the columns TransactionId, timestamp, orginLocation, destLocation as they have too many unique categorical values which can't be converted into numbers easily.
#2. Fill the null values of originCountry
#3. We will encode type, currency, originBank, destBank, originCountry, destCountry as they have less no of unique values.
#4. Amount is a column which has to be in int or float datatype but it is an object right now we need to change that.
#5. Handling the float64 and int columns is the easiest, don't do anything.
    #executing step 1
    drop_columns = ['transactionId', 'timestamp', 'originLocation', 'destLocation']
    initial_dataset.drop(drop_columns, axis=1, inplace=True)
    #executing step 2
    #originCountry is a categorical column which means we can't take mean or meadian of the column to replace it with the null values. Therefore we will use UNKNOWN
    cols_missing = ['originCountry']
    for col in cols_missing:
        if col in initial_dataset.columns:
                initial_dataset[col] = initial_dataset[col].fillna("UNKNOWN")
    #saving the label for test set
    labels = initial_dataset['is_anomaly']
    initial_dataset.drop('is_anomaly', axis=1, inplace=True)
    #executing step 4
    initial_dataset['amount']= initial_dataset['amount'].astype(str).str.replace(',','').astype(float)
    initial_dataset['originBank'] = initial_dataset['originBank'].str.upper()
    #executing step 3
    cat_cols = ['type', 'currency', 'originBank', 'originCountry', 'destBank', 'destCountry']
    num_cols = ['amount', 'oldBalanceOrg', 'newBalanceOrg', 'oldBalanceDest', 'newBalanceDest']
    df_encoded= pd.get_dummies(initial_dataset[cat_cols], drop_first=True)
    scaler= StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(initial_dataset[num_cols])
    #preprocessing done now
    df_encoded = df_encoded.astype(float)
    X= df_encoded.copy()
    normal_data = X[labels==0]
    return torch.FloatTensor(normal_data.values)