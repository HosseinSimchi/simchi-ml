##### میتونیم فرآیند یادگیری ماشین رو به قسمت های مختلفی دسته بندی کرد

###### Create three files --> `split.py`, `preprocess.py`, `train.py`

###### Separate all sections from the `app.py` and add inside the created files

- ###### `src/split.py` --> create temp_data folder and add .gitignore(add \*\* inside the file) inside that folder

  ```py
  # چون از روت این کدهارو اجرا میکنیم نیازه که مسیر فایل هارو طوری بدهیم که از روت قابل دسترسی هستند

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
  ```

- ###### `src/preprocess.py`

  ```py
  # چون از روت این کدهارو اجرا میکنیم نیازه که مسیر فایل هارو طوری بدهیم که از روت قابل دسترسی هستند
  import pandas as pd
  import numpy
  from sklearn.preprocessing import StandardScaler

  X_train = pd.read_csv('temp_data/X_train.csv')
  X_test = pd.read_csv('temp_data/X_test.csv')

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # We use savetxt because the X_train and X_test are nparray
  numpy.savetxt('temp_data/X_train.csv',X_train, delimiter=",")
  numpy.savetxt('temp_data/X_test.csv',X_test, delimiter=",")
  ```

- ###### `src/train.py`

  ```py
  import pandas as pd
  import tensorflow as tf
  from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
  from tensorflow.keras.models import Sequential, load_model
  from tensorflow.keras.layers import Dense
  from sklearn.metrics import accuracy_score

  X_train = pd.read_csv('temp_data/X_train.csv', header=None)
  X_test = pd.read_csv('temp_data/X_test.csv', header=None)

  y_train = pd.read_csv('temp_data/y_train.csv')
  y_test = pd.read_csv('temp_data/y_test.csv')

  model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
  lr_schedule = LearningRateScheduler(lambda epoch: 1e-3 * 0.9 ** epoch)

  history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
                      callbacks=[early_stopping, lr_schedule])

  test_loss, test_accuracy = model.evaluate(X_test, y_test)
  print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

  model.save('models/tfmodel.keras')
  ```
