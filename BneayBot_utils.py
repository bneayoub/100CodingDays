from tqdm import tqdm
import os
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import html
import tensorflow as tf
from matplotlib import pyplot as plt

def pull_twitter(twitter_filepath, shuffle=True):
  with open(twitter_filepath, "r", encoding="utf-8") as twt_f:
    lines = twt_f.read().split("\n")

  inputs, outputs = list(), list()
  for i, l in enumerate(tqdm(lines)):
    if i % 2 == 0:
      inputs.append(bytes(html.unescape(l).lower(), "utf-8"))
    else:
      outputs.append(bytes(html.unescape(l).lower(), "utf-8"))

  popped = 0
  for i, (ins, outs) in enumerate(zip(inputs, outputs)):
    if not ins or not outs:
      ins.pop(i)
      outs.pop(i)
      popped += 1

  print(f"Pairs popped: {popped}")
  if shuffle:
    print("\nShuffling...")
    inputs, outputs = shuffle_inputs_outputs(inputs, outputs)

  return inputs, outputs

def shuffle_inputs_outputs(inputs, outputs):
  inputs_outputs = list(zip(inputs, outputs))
  random.shuffle(inputs_outputs)
  inputs, outputs = zip(*inputs_outputs)
  return inputs, outputs