import json
import os

TRAIN, EVAL, PREDICT = "train eval predict".split()
MODES = [TRAIN, EVAL, PREDICT]


def get_features_and_labels(data_dir, get_labels=False):

  features = []
  texts = []
  identifiers = []
  with open(f"{data_dir}/features.jsonl", "r") as f:
    for line in f:
      example = json.loads(line)
      features.append(example["features"])
      texts.append(example["text"])
      identifiers.append(example["identifier"])
  if get_labels:
    labels = []
    with open(f"{data_dir}/labels.jsonl", "r") as f:
      for i, line in enumerate(f):
        example = json.loads(line)
        assert example["identifier"] == identifiers[i]
        labels.append(example["label"])
  else:
    labels = None

  return identifiers, features, texts, labels
