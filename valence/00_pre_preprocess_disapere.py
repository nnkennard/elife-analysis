import argparse
import glob
import json
import os

import preprocess_lib

parser = argparse.ArgumentParser(description="Extract DISAPERE data")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to data file containing score jsons",
)


def main():
  # Make output dir if required

  args = parser.parse_args()

  for subset in "train dev test".split():
    output_dir = f"{args.output_dir}/disapere_{subset}/"
    os.makedirs(output_dir, exist_ok=True)
    sentences = []
    labels = []
    for filename in glob.glob(f"{args.data_dir}/final_dataset/{subset}/*.json"):
      obj = preprocess_lib.get_json_obj(filename)
      review_id = obj["metadata"]["review_id"]
      for i, sentence in enumerate(obj["review_sentences"]):
        identifier = preprocess_lib.make_identifier(review_id, i)
        sentences.append({"identifier": identifier, "text": sentence["text"]})
        labels.append({
            "identifier": identifier,
            "label": 0 if sentence["pol"] == "none" else 1,
        })

    with open(f"{output_dir}/sentences.jsonl", "w") as f:
      for s in sentences:
        f.write(json.dumps(s) + "\n")
    with open(f"{output_dir}/labels.jsonl", "w") as f:
      for l in labels:
        f.write(json.dumps(l) + "\n")


if __name__ == "__main__":
  main()
