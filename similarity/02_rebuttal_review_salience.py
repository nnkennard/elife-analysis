import argparse
import collections
import json
import tqdm

import similarity_lib

parser = argparse.ArgumentParser(description="Prepare tokenized paper text")
parser.add_argument(
    "-o",
    "--output_dir",
    default="disapere_results/",
    type=str,
    help="output file",
)

def main():

  args = parser.parse_args()

  with open(f'{args.output_dir}/similarity_tokenized.json', "r") as f:
    input_data = json.load(f)

  results = []
  for structure in tqdm.tqdm(input_data["structures"]):
    for review in structure["reviews"]:
      review_sentences = review["review_sentences"]
      rebuttal_sentences = review["rebuttal_sentences"]
      results.append({
          "review_id": review["review_id"],
          "review_sentences": review_sentences,
          "rebuttal_sentences": rebuttal_sentences,
          "result": {
              fn_name: fn(review_sentences, rebuttal_sentences)
              for fn_name, fn in similarity_lib.FUNCTION_MAP.items()
          },
      })

  with open(f'{args.output_dir}/rebuttal_review_salience.json', "w") as f:
    json.dump(results, f)


if __name__ == "__main__":
  main()
