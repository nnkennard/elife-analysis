import argparse
import glob
import json
import os

import preprocess_lib

def polarity_exists(sentence):
    return 0 if sentence["polarity"] == "none" else 1


LABEL_EXTRACTORS = {
        "polarity_exists": polarity_exists,
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
    "-l",
    "--label_extractor",
    choices=LABEL_EXTRACTORS.keys(),
    help="name of the label extractor to apply to each sentence",
)

def main():

  args = parser.parse_args()
  if args.label_extractor is None:
      print("Needs a label extractor")
      exit()

  for subset in "train dev test".split():
    output_dir = f"{args.output_dir}/{args.label_extractor}/{subset}/"
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
            "label": LABEL_EXTRACTORS[args.label_extractor](sentence),
        })

    with open(f"{output_dir}/sentences.jsonl", "w") as f:
      for s in sentences:
        f.write(json.dumps(s) + "\n")
    with open(f"{output_dir}/labels.jsonl", "w") as f:
      for l in labels:
        f.write(json.dumps(l) + "\n")


if __name__ == "__main__":
  main()
