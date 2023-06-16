import os
import numpy as np
import librosa

def create_dataset(audio_files):
  """Creates a dataset of audio files and their corresponding beatmaps.

  Args:
    audio_files: A list of paths to audio files.

  Returns:
    A list of dictionaries, where each dictionary maps an audio file to its corresponding beatmap.
  """

  dataset = []
  for audio_file in audio_files:
    try:
      rate, audio = librosa.load(audio_file, sr=16000)
      beatmap = []
      dataset.append({'rate': rate, 'audio': audio, 'beatmaps': beatmap})
    except FileNotFoundError:
      print(f"FileNotFoundError: {audio_file}")

  return dataset
