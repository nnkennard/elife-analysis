import argparse
import collections
import itertools
import json
import tqdm

import similarity_lib

parser = argparse.ArgumentParser(description="Prepare tokenized paper text")
parser.add_argument(
    "-i",
    "--input_file",
    default="disapere_similarity_full.json",
    type=str,
    help="input json file",
)
parser.add_argument(
    "-o",
    "--output_dir",
    default="disapere_results/",
    type=str,
    help="output file",
)


def main():

  args = parser.parse_args()

  with open(args.input_file, "r") as f:
    input_data = json.load(f)

  results = []
  for structure in tqdm.tqdm(input_data["structures"]):
    if len(structure["reviews"]) == 1:
      continue
    for review_1, review_2 in itertools.combinations(structure["reviews"], 2):
      review_1_sentences = review_1["tokenized_review"]
      review_2_sentences = review_2["tokenized_review"]
      results.append({
          "review_1_id": review_1["review_id"],
          "review_2_id": review_2["review_id"],
          "review_1_sentences": review_1_sentences,
          "review_2_sentences": review_2_sentences,
          "result": {
              fn_name: fn(review_1_sentences, review_2_sentences)
              for fn_name, fn in similarity_lib.FUNCTION_MAP.items()
          },
      })

  with open(f'{args.output_dir}/review_review_consonance.json', "w") as f:
    json.dump(results, f)


if __name__ == "__main__":
  main()
