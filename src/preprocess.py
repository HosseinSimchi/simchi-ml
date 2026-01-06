import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('temp_data/X_train.csv')
X_test = pd.read_csv('temp_data/X_test.csv')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

numpy.savetxt('temp_data/X_train_pre.csv',X_train, delimiter=",")
numpy.savetxt('temp_data/X_test_pre.csv',X_test, delimiter=",")