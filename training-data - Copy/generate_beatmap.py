import tensorflow as tf
import librosa
import beatmap
import os

import train_model



def generate_beatmap(audio_file):
  """Generates a beatmap for the given audio file.

  Args:
    audio_file: A path to an audio file.

  Returns:
    A beatmap for the audio file.
  """

  model = train_model.load_model()

  beatmap = model.predict(audio_file)

  return beatmap

def write_beatmap(beatmap, filename):
  """Writes the given beatmap to a .osu file.

  Args:
    beatmap: A beatmap.
    filename: The path of the .osu file to write to.
  """

  with open(filename, 'w') as f:
    f.write(beatmap.to_osu())

if __name__ == "__main__":
  audio_file = "summer-walk.wav"

  beatmap = generate_beatmap(audio_file)

  write_beatmap(beatmap, 'summer-walk.osu')
