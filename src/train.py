import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import json
import os
from pathlib import Path

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
  
model_dir = 'models'
Path(model_dir).mkdir(parents=True, exist_ok=True)
model.save('models/tfmodel.keras')