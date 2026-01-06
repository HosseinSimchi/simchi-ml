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
  # عمدا با نام متفاوتی ذخیره میکنیم که بعدا در تعریف استیج به مشکل برنخوریم
  numpy.savetxt('temp_data/X_train_pre.csv',X_train, delimiter=",")
  numpy.savetxt('temp_data/X_test_pre.csv',X_test, delimiter=",")
  ```

- ###### `src/train.py`

  ```py
  import pandas as pd
  import tensorflow as tf
  from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
  from tensorflow.keras.models import Sequential, load_model
  from tensorflow.keras.layers import Dense
  from sklearn.metrics import accuracy_score
  import json

  X_train = pd.read_csv('temp_data/X_train_pre.csv', header=None)
  X_test = pd.read_csv('temp_data/X_test_pre.csv', header=None)

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

  metrics = {
    "test_loss": test_loss,
    "test_accuracy": test_accuracy
  }

  with open("temp_data/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

  model.save('models/tfmodel.keras')
  ```

###### ADD stages to DVC

- ###### 1- `python3 -m dvc stage add -n split -d data/cleaned_dataset.csv -d src/split.py -o temp_data/X_train.csv -o temp_data/X_test.csv -o temp_data/y_train.csv -o temp_data/y_test.csv python3 src/split.py`

  - ###### After running this command, the `dvc.yml` will be generated.
  - ###### -n `split` is the custom name for the first stage
  - ###### -d `data/cleaned_dataset.csv` is the path of dataset for input of the file
  - ###### -d `src/split.py` is the input .py file

- ###### 2- `python3 -m dvc stage add -n preprocessing -d temp_data/X_train.csv -d temp_data/X_test.csv -d src/preprocess.py -o temp_data/X_train_pre.csv -o temp_data/X_test_pre.csv python3 src/preprocess.py`

- ###### 3- `python3 -m dvc stage add -n train -d temp_data/X_train_pre.csv -d temp_data/X_test_pre.csv -d temp_data/y_train.csv -d temp_data/y_test.csv -d src/train.py -o models/tfmodel.keras -M temp_data/metrics.json python3 src/train.py`
  - ###### -M metrics.json --> متریک هارو ذخیره میکنه

- ###### 4- `python -m dvc repro`
- ###### 5- Everyone pull the changes should do the previous command
- ###### 6- استج هایی که تغییری نکرده اند در صورت ران شدن دستور شماره چهارم تغییری ایجاد نمیشه یا به عبارتی دیگر در هربار اجرای دستورچهارم تمامی استیت ها کش می شوند
- ###### 7- Add the fourth command to the `ci.yml` to do these steps when user push codes on the GITHUB (TO the Train model stage)