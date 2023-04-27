import argparse
import collections
import csv

import who_wins_lib

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

parser = argparse.ArgumentParser(description="Analyze Who Wins results")
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="path to overall data directory",
)
parser.add_argument(
    "-f",
    "--filter_config",
    type=str,
    help="path to configs of results to filter by",
)
parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    help="confidence threshold for filtering",
)

Result = collections.namedtuple("Result", "precision recall fscore support".split())

def modify_with_confidence(probabilities, positive_class, confidence_threshold):
  labels = []
  for p_array in probabilities:
    #print("!!", p_array[positive_class])
    if max(p_array) == p_array[positive_class]:
      print("!!", p_array[positive_class])
    if p_array[positive_class] > confidence_threshold and max(p_array) == p_array[positive_class]:
      labels.append(1)
    else:
      labels.append(0)
  return labels

def modify_to_binary(labels, positive_class):
  return [
    int(label == positive_class)  for label in labels
  ]

def main():
    args = parser.parse_args()
    config = who_wins_lib.read_config(args.config)

    for source in config.dev:
        with open(f"results/{config.config_name}_dev.csv", "r") as f:
            reader = csv.DictReader(f)
            label_names = reader.fieldnames[3:]
            rows = [r for r in reader]

            predictions = [int(row['label']) for row in rows]
            true_labels = [int(row['true_label']) for row in rows]
            probabilities = [[float(row[name]) for name in label_names] for row in rows]
            #true_class_probs = [
            #  row[label_names[int(row['true_label'])]] for row in rows
            #]
            #top_class_probs = [
            #  row[label_names[int(row['label'])]] for row in rows
            #]

    print(collections.Counter(predictions))
    print(collections.Counter(true_labels))


    #predictions = predictions[:10]
    #true_labels = true_labels[:10]
    #probabilities = probabilities[:10]



    for label_i, label in enumerate(label_names):
      new_labels = modify_to_binary(true_labels, label_i)
      for confidence_x100 in range(0, 1000, 100):
        confidence = confidence_x100 / 1000
        print(label, confidence)
        new_predictions = modify_with_confidence(probabilities, label_i, confidence)
        print(new_labels)
        print(new_predictions)
        metrics = precision_recall_fscore_support(true_labels, predictions)
        print(metrics[1][label_i])
        #print(len(metrics), len(metrics[0]))
        #print(metrics)
        print()
      print()


    #for k, v in results.items():
    #  for kk, vv in v.items():
    #    print(k, kk, vv)
    #  print()




if __name__ == "__main__":
    main()
