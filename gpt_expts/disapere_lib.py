import collections
import csv
import glob
import json

DISAPERE_DIR = "../data/raw/disapere/"
collections

def make_identifier(sentence_dict):
  return sentence_dict["review_id"] + "|||" + str(
      sentence_dict["sentence_index"])


def extract_review_examples(filename, label):
  examples = []
  with open(filename, "r") as f:
    obj = json.load(f)
    for sentence in obj["review_sentences"]:
      examples.append((
          make_identifier(sentence),
          sentence["text"].replace("\t", " "),
          sentence[label],
      ))
  return examples


def get_dataset(task):
  assert task in "review_action fine_review_action aspect polarity".split(
  )

  examples = collections.defaultdict(lambda:collections.defaultdict(list))
  for subset in "train dev test".split():
    for filename in glob.glob(
          f"{DISAPERE_DIR}/{subset}/*"):
      for identifier, text, label in extract_review_examples(
            filename, task):
           examples[subset][label].append((identifier, text))
  return examples
          
