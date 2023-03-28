import argparse
import collections
import csv

import who_wins_lib

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

Results = collections.namedtuple("Results", "tp tn fp fn".split())


def calculate_metrics(predictions, true_labels, label_names):
    if len(label_names) == 2:
        return metrics(categorize_binary_labels(predictions, true_labels))
    else:
        results = {}
        for label_i, label_name in enumerate(label_names):
            binary_predictions = [pred == label_i for pred in predictions]
            binary_labels = [t == label_i for t in true_labels]
            results.update(
                metrics(
                    categorize_binary_labels(binary_labels, binary_predictions),
                    label_name,
                )
            )

        f1s = [v for k, v in results.items() if 'f1' in k]
        assert len(f1s) == len(label_names)
        results["macro_f1"] = sum(f1s)/len(f1s)
        return results


def categorize_binary_labels(predictions, true_labels):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for prediction, true_label in zip(predictions, true_labels):
        if true_label:
            if prediction:
                tp += 1
            else:
                fn += 1
        else:
            if prediction:
                fp += 1
            else:
                tn += 1

    return Results(tp=tp, tn=tn, fp=fp, fn=fn)


def metrics(results, name="bin"):
    total = sum(results)
    if results.tp or results.fp:
        precision = results.tp / (results.tp + results.fp)
    else:
        precision = 0.0
    if results.tp or results.fn:
        recall = results.tp / (results.tp + results.fn)
    else:
        recall = 0.0

    if precision or recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        f"accuracy_{name}": (results.tp + results.tn) / total,
        f"precision_{name}": precision,
        f"recall_{name}": recall,
        f"f1_{name}": f1,
    }


def main():
    args = parser.parse_args()
    config = who_wins_lib.read_config(args.config)

    for source in config.dev:
        with open(f"results/{config.config_name}_dev.csv", "r") as f:
            reader = csv.DictReader(f)
            label_names = reader.fieldnames[3:]
            predictions = []
            true_labels = []
            for row in reader:
                predictions.append(int(row["label"]))
                true_labels.append(int(row["true_label"]))
        results = calculate_metrics(predictions, true_labels, label_names)
        for k, v in results.items():
            print(k, v)


if __name__ == "__main__":
    main()
