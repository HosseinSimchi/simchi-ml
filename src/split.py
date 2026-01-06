import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/cleaned_dataset.csv')
X = df[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']]
y = df['Gesture']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

X_train.to_csv('temp_data/X_train.csv', index_label=False, index=False)
X_test.to_csv('temp_data/X_test.csv', index_label=False, index=False)

y_train.to_csv('temp_data/y_train.csv', index_label=False, index=False)
y_test.to_csv('temp_data/y_test.csv', index_label=False, index=False)