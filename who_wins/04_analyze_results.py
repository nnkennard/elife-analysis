import argparse
import collections
import csv
import io
import yaml
import who_wins_lib

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# from sklearn.metrics import f1_score

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
parser.add_argument(
    "-ec",
    "--epi_config",
    type=str,
    help="config name of final epi predictions",
)

Config = collections.namedtuple(
    "Config", "config_name task model_name train dev test predict labels".split()
)


def read_config(config_name, schema_path="schema.yml"):
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(io.StringIO(f.read()))

    with open(f"{CONFIG_PATH}/{config_name}.yml", "r") as f:
        config = yaml.safe_load(io.StringIO(f.read()))
        assert config["config_name"] == config_name
        assert config["task"] in schema["tasks"]
        return Config(labels=schema["labels"][config["task"]], **config)
    
Result = collections.namedtuple("Result", "precision recall fscore support".split())


def modify_with_confidence(probabilities, positive_class, confidence_threshold):
    labels = []
    for p_array in probabilities:
        if (
            p_array[positive_class] > confidence_threshold
            and max(p_array) == p_array[positive_class]
        ):
            labels.append(1)
        else:
            labels.append(0)
    return labels
    
    
def modify_to_binary(labels, positive_class):
    return [int(label == positive_class) for label in labels]


ResultFile = collections.namedtuple(
    "ResultFile", "predictions true_labels probabilities label_names"
)


# def read_result_file(filename, source):
#     with open(filename, "r") as f:
#         reader = csv.DictReader(f)
#         label_names = reader.fieldnames[3:]
#         rows = [r for r in reader]
#     predictions = [int(row["label"]) for row in rows if source in row['identifier']]
#     true_labels = [int(row["true_label"]) for row in rows if source in row['identifier']]
#     probabilities = [[float(row[name]) for name in label_names] for row in rows if source in row['identifier']]

#     return ResultFile(predictions, true_labels, probabilities, label_names)


def read_result_file(filename, source, epi_config, confidence_threshold):
    # when not analyzing epi labels
    # get review ids where we are confident sent is epi
    if epi_config: 
        with open(f"results/{epi_config}_dev.csv") as f:
            reader = csv.DictReader(f)
            ids = [r['identifier'] for r in reader if float(r['epi'])>=confidence_threshold]
            print(f"Subsetting to {len(ids)} labels that have confidence of {confidence_threshold}.")
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            label_names = reader.fieldnames[3:]
            # subset to confidently epi ids
            rows = [r for r in reader if r['identifier'] in ids]
    else:
        reader = csv.DictReader(f)
        label_names = reader.fieldnames[3:]
        rows = [r for r in reader]        
    predictions = [int(row["label"]) for row in rows if source in row['identifier']]
    true_labels = [int(row["true_label"]) for row in rows if source in row['identifier']]
    probabilities = [[float(row[name]) for name in label_names] for row in rows if source in row['identifier']]
        
    return ResultFile(predictions, true_labels, probabilities, label_names)

def main():
    args = parser.parse_args()
    config = who_wins_lib.read_config(args.config)

    for source in config.dev:
        # You can also do this on train if you have run eval with the train set
        result_file = read_result_file(f"results/{config.config_name}_dev.csv", 
                                       source,
                                       args.epi_config, 
                                       float(args.threshold))
        unused_confidence_stuff = """
        for confidence_x100 in range(0, 1000, 100):
            confidence = confidence_x100 / 1000
            for label_i, label in enumerate(result_file.label_names):
                new_labels = modify_to_binary(result_file.true_labels, label_i)
                new_predictions = modify_with_confidence(
                   result_file.probabilities, label_i, confidence
                )
                metrics = precision_recall_fscore_support(
                  result_file.true_labels, result_file.predictions)
                print(label, confidence, metrics)
            print()
        """
        # metrics = precision_recall_fscore_support(
        metrics = precision_recall_fscore_support(
            result_file.true_labels, result_file.predictions
        )
        print(source, " ".join(result_file.label_names))
        for name, metric in zip("precision recall fscore support".split(),
        metrics):
            print(name, metric)
        print()



if __name__ == "__main__":
    main()
