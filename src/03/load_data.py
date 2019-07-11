import tensorflow as tf
import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler # This used for scaling our data
CURRECTPATH = os.getcwd() + '/resources/'
print(os.getcwd())
#/Users/Ghasak/Desktop/MPDATA/LearningMB/11_Machine_Learning_Lyinda_tensorflow/resources/sales_data_test.csv
traning_data_df = pd.read_csv(CURRECTPATH + 'sales_data_training.csv', dtype = float)
print(traning_data_df.head())

# Pull out columns for X (data to train with) and Y (value to predict)
X_training = traning_data_df.drop('total_earnings', axis= 1).values # .value will make it an array
Y_training = traning_data_df[['total_earnings']].values

# Load testing data set from CSV file

test_data_df = pd.read_csv(CURRECTPATH +'sales_data_test.csv', dtype = float)

# Pull out columns for X (Data to train with) and Y (value to predict)

X_testing = test_data_df.drop('total_earnings', axis = 1).values
Y_testing = test_data_df[['total_earnings']].values



# All data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well. Create scalers for teh inputs and outups
X_scaler = MinMaxScaler(feature_range=(0,1))
Y_scaler = MinMaxScaler(feature_range=(0,1))


# Scale both the traning inputs and outputs
X_scaled_traning = X_scaler.fit_transform(X_training)
Y_scaled_traning = Y_scaler.fit_transform(Y_training)

# It's very import that the traning and test data are scaled with the same scaler

X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)


print("Note: Y value were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0],Y_scaler.min_[0]))
