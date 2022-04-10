import argparse
import collections
import json
import tqdm

import preprocess_lib

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)


def main():

  args = parser.parse_args()

  results_by_review_id = collections.defaultdict(lambda:
  collections.defaultdict(dict))

  for model in "bert dep".split():
    with open(
    f'{args.data_dir}/{model}_predictions.jsonl', 'r'
    ) as f:
      for line in f:
        obj = json.loads(line)
        print(obj)
        review_id, index = preprocess_lib.split_identifier(obj['identifier'])
        results_by_review_id[review_id][index][model] = obj['label']

  text_dict = collections.defaultdict(dict)
  with open(
  f'{args.data_dir}/features.jsonl', 'r'
  ) as f:
    for line in f:
      obj = json.loads(line)
      print(obj)
      review_id, index = preprocess_lib.split_identifier(obj['identifier'])
      text_dict[review_id][index] = obj['text']

  for review_id, sentence_texts in tqdm.tqdm(text_dict.items()):
    key_list = list(sentence_texts.keys())
    assert list(sorted(key_list)) == list(range(len(key_list)))
    for i in range(len(key_list)):
      nps = preprocess_lib.get_maximal_nps(sentence_texts[i])
      print(results_by_review_id[review_id][i])
      print(nps)


if __name__ == "__main__":
  main()

