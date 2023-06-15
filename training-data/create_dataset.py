import os
import numpy as np
import librosa

def create_dataset(audio_files, beatmaps):
  """Creates a dataset of audio files and their corresponding beatmaps.

  Args:
    audio_files: A list of paths to audio files.
    beatmaps: A list of paths to beatmaps.

  Returns:
    A dictionary of audio files to beatmaps.
  """

  dataset = {}
  for audio_file, beatmap in zip(audio_files, beatmaps):
    dataset[audio_file] = beatmap

  return dataset

if __name__ == "__main__":
  audio_files = []
  beatmaps = []

  for file in os.listdir("./audio"):
    if file.endswith(".mp3"):
      audio_files.append("./audio/" + file)

  for file in os.listdir("./beatmaps"):
    if file.endswith(".osu"):
      beatmaps.append("./beatmaps/" + file)

  dataset = create_dataset(audio_files, beatmaps)

  np.save("dataset.npy", dataset)
