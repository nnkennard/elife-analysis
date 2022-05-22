import argparse
import collections
import json
import tqdm

import similarity_lib

parser = argparse.ArgumentParser(description="Review-manuscript salience")
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
    default="disapere_review_manuscript_salience.json",
    type=str,
    help="output file",
)


def main():

  args = parser.parse_args()

  with open(args.input_file, "r") as f:
    input_data = json.load(f)

  manuscript_text_map = {
    x["forum_id"]:sum(x["tokenized_manuscript"].values(),[]) for x in
    input_data["tokenized_manuscripts"]
  }

  results = []
  for structure in tqdm.tqdm(input_data["structures"][:3]):
    for review in structure["reviews"]:
      review_sentences = review["tokenized_review"]
      manuscript_sentences = manuscript_text_map.get(structure["forum_id"], None)
      if manuscript_sentences is None:
        continue
      results.append({
          "review_id": review["review_id"],
          "review_sentences": review_sentences,
          "manuscript_sentences": manuscript_sentences,
          "result": {
              fn_name: fn(review_sentences, manuscript_sentences)
              for fn_name, fn in similarity_lib.FUNCTION_MAP.items()
          },
      })

  with open(args.output_file, "w") as f:
    json.dump(results, f)


if __name__ == "__main__":
  main()
