import argparse
import collections
import json
import joblib
import pickle
import tqdm

import classification_lib

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    help="train eval or predict",
)

DEP_CLASSIFIER_PATH = "ckpt/dep_classifier.joblib"

examples = collections.defaultdict(list)
labels = collections.defaultdict(list)


def unused_do_analyze():
  (_, vectorizer), (_, classifier) = pipe.steps
  sorted_features = sorted(
      zip(vectorizer.get_feature_names_out(), classifier.coef_[0]),
      key=lambda x: -1 * x[1]**2,
  )

  print("Influential features")
  for i, (a, b) in enumerate(sorted_features):
    if i == 10:
      break
    print(i, "-".join(eval(a)), b)


def do_train(data_dir):

  assert "train" in data_dir

  _, features, _, labels = classification_lib.get_features_and_labels(
      data_dir, get_labels=True)

  pipe = Pipeline([
      ("vectorizer", DictVectorizer(sparse=False)),
      ("clf", LogisticRegression(random_state=0, max_iter=1500)),
  ])
  pipe.fit(features, labels)
  print("Train accuracy", pipe.score(features, labels))
  joblib.dump(pipe, DEP_CLASSIFIER_PATH)


def do_eval(data_dir):
  assert "dev" in data_dir
  _, features, _, labels = classification_lib.get_features_and_labels(
      data_dir, get_labels=True)
  pipe = joblib.load(DEP_CLASSIFIER_PATH)
  print("Dev accuracy", pipe.score(features, labels))


def do_predict(data_dir):
  identifiers, features, _, _ = classification_lib.get_features_and_labels(
      data_dir, get_labels=False)
  pipe = joblib.load(DEP_CLASSIFIER_PATH)
  with open(f"{data_dir}/dep_predictions.jsonl", "w") as f:
    for identifier, label in zip(identifiers, pipe.predict(features)):
      f.write(
          json.dumps({
              "identifier": identifier,
              "label": int(label),
          }) + "\n")


def main():

  args = parser.parse_args()
  assert args.mode in classification_lib.MODES

  if args.mode == classification_lib.TRAIN:
    do_train(args.data_dir)
  elif args.mode == classification_lib.EVAL:
    do_eval(args.data_dir)
  elif args.mode == classification_lib.PREDICT:
    do_predict(args.data_dir)


if __name__ == "__main__":
  main()
