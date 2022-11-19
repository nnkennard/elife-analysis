import json
import os

TRAIN, EVAL, PREDICT = "train eval predict".split()
MODES = [TRAIN, EVAL, PREDICT]


def get_text_and_labels(data_dir, get_labels=False):

  texts = []
  identifiers = []
  with open(f"{data_dir}/sentences.jsonl", "r") as f:
    for line in f:
      example = json.loads(line)
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

  return identifiers, texts, labels
