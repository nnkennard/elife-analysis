import argparse
import collections
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
    "--output_file",
    default="disapere_rebuttal_review_salience.json",
    type=str,
    help="output file",
)



def main():

  args = parser.parse_args()


  with open(args.input_file, "r") as f:
    input_data = json.load(f)

  results = []
  for structure in tqdm.tqdm(input_data["structures"]):
    for review in structure["reviews"]:
      review_sentences = review['tokenized_review']
      rebuttal_sentences = review['tokenized_rebuttal']
      results.append({
          "review_id": review["review_id"],
          "review_sentences": review_sentences,
          "rebuttal_sentences": rebuttal_sentences,
          "result": {fn_name: fn(review_sentences, rebuttal_sentences) for
          fn_name, fn in similarity_lib.FUNCTION_MAP.items()}
      })

  with open(args.output_file, "w") as f:
    json.dump(results, f)


if __name__ == "__main__":
  main()
