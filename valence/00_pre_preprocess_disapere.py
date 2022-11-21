import argparse
import collections
import glob
import json
import os

import preprocess_lib


def polarity_exists(sentence):
    return 0 if sentence["polarity"] == "none" else 1


def review_action(sentence):
    return sentence["review_action"]


TASK_MAP = {
    "polarity_exists": polarity_exists,
    "review_action": review_action,
}

parser = argparse.ArgumentParser(description="Extract DISAPERE data")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to main DISAPERE directory (should contain final_dataset/ as a subdirectory)",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to output directory (will be created if necessary)",
)
parser.add_argument(
    "-t",
    "--task",
    choices=TASK_MAP.keys(),
    help="name of the label extractor to apply to each sentence",
)


def main():

    args = parser.parse_args()

    sentences = collections.defaultdict(list)
    for subset in "train dev test".split():
        for filename in glob.glob(f"{args.data_dir}/final_dataset/{subset}/*.json"):
            obj = preprocess_lib.get_json_obj(filename)
            review_id = obj["metadata"]["review_id"]
            for i, sentence in enumerate(obj["review_sentences"]):
                identifier = preprocess_lib.make_identifier(review_id, i)
                sentences[subset].append(
                    (
                        identifier,
                        sentence["text"],
                        TASK_MAP[args.task](sentence),
                    )
                )

    task_output_dir = f"{args.output_dir}/{args.task}/"
    os.makedirs(task_output_dir, exist_ok=True)
    label_map = list(sorted(set(t[2] for t in sum(sentences.values(), []))))
    with open(f"{task_output_dir}/metadata.json", "w") as f:
        json.dump({"labels": label_map}, f)

    for subset, subset_sentences in sentences.items():
        with open(f'{task_output_dir}/{subset}.jsonl', 'w') as f:
            for identifier, text, label in subset_sentences:
                f.write(
                    json.dumps(
                        {"identifier": identifier,
                        "text":text,
                        "label": label_map.index(label)}
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
