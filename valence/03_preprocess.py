import argparse
import glob
import json
import os
import tqdm

import preprocess_lib

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-u",
    "--unsplit_data_dir",
    type=str,
    help="path to data file containing score jsons",
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)
parser.add_argument(
    "-s",
    "--skip_split",
    action="store_true",
    help="whether to skip sentence split",
)


def get_sentences(review_obj):
  sentence_list = []
  for i, sentence in enumerate(
      preprocess_lib.SENTENCIZE_PIPELINE(review_obj["review_text"]).sentences):
    sentence_list.append({
        "identifier":
            preprocess_lib.make_identifier(review_obj["review_id"], i),
        "text":
            sentence.text,
    })
  return sentence_list


def main():

  args = parser.parse_args()

  if not args.skip_split:
    examples = []

    for filename in glob.glob(f"{args.unsplit_data_dir}/*.json"):
      reviews = preprocess_lib.get_json_obj(filename)
      for review in reviews:
        examples += get_sentences(review)

    os.makedirs(f"{args.data_dir}/", exist_ok=True)
    with open(f"{args.data_dir}/sentences.jsonl", "w") as f:
      for e in examples:
        f.write(json.dumps(e) + "\n")

  with open(f"{args.data_dir}/sentences.jsonl", "r") as f:
    examples = [json.loads(line) for line in f]

  with open(f"{args.data_dir}/features.jsonl", "w") as f:
    for example in tqdm.tqdm(examples):
      # It's not ideal to do this on a sentence-by-sentence basis (too slow)
      # But it only needs to happen once per dataset, and I think it's cleaner
      # to deal with everything as separate sentences.
      example["features"] = preprocess_lib.featurize_sentence(example["text"])
      f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
  main()
