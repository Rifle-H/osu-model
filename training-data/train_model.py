import os
import numpy as np
import tensorflow as tf

def train_model(dataset):
  """Trains a model on the given dataset.

  Args:
    dataset: A dictionary of audio files to beatmaps.

  Returns:
    A trained model.
  """

  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(dataset['audio'].shape[1], dataset['audio'].shape[2])),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(dataset['beatmaps'])),
  ])

  model.compile(optimizer='adam', loss='mse')

  model.fit(dataset['audio'], dataset['beatmaps'], epochs=10)

  return model

if __name__ == "__main__":
  dataset = np.load("dataset.npy", allow_pickle=True).item()

  model = train_model(dataset)

  model.save("model.h5")
