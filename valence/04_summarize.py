import argparse
import collections
import json
import os
import tqdm

import preprocess_lib

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)

html_template = """
<HTML>
<head>
    <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css">
    </head>
    <body>
    <div class="container">
    {0}
    </div>
    </body>
    </HTML>
    """


def build_table_from_rows(rows):
  table_template = """
  <table class="table">
  <thead>
  <tr>
  <th> Index </th>
  <th> Sentence </th>
  <th> Polarity label (BERT) </th>
  <th> Polarity label (Deps) </th>
  <th> Maximal NPs </th>
  </tr>
  </thead>
  <tbody>
  {0}
  </tbody>
  </table>
  """
  row_texts = []
  for row in rows:
    row_texts.append("""
  <tr>
  <td> {0} </td>
  <td> {1}</td> 
  <td> {2} </td> 
  <td> {3} </td> 
  <td> {4} </td> 
  </tr>  """.format(row["idx"], row["sentence"], row["bert"], row["dep"],
                    row["nps"]))
  return table_template.format("\n".join(row_texts))


def build_index(keys):
  index_text = "<ol>"
  for key in keys:
    index_text += f'<li><a href="{key}.html">{key}</a></li>'
  index_text += "</ol>"
  return index_text


def main():

  args = parser.parse_args()

  results_by_review_id = collections.defaultdict(
      lambda: collections.defaultdict(dict))

  for model in "bert dep".split():
    with open(f"{args.data_dir}/{model}_predictions.jsonl", "r") as f:
      for line in f:
        obj = json.loads(line)
        print(obj)
        review_id, index = preprocess_lib.split_identifier(obj["identifier"])
        results_by_review_id[review_id][index][model] = obj["label"]

  text_dict = collections.defaultdict(dict)
  with open(f"{args.data_dir}/features.jsonl", "r") as f:
    for line in f:
      obj = json.loads(line)
      review_id, index = preprocess_lib.split_identifier(obj["identifier"])
      text_dict[review_id][index] = obj["text"]

  os.makedirs(f'{args.data_dir}/viewer/', exist_ok=True)

  for review_id, sentence_texts in tqdm.tqdm(text_dict.items()):
    key_list = list(sentence_texts.keys())
    assert list(sorted(key_list)) == list(range(len(key_list)))
    rows = []
    for i in range(len(key_list)):
      rows.append({
      "idx": i,
      "sentence": sentence_texts[i],
      "nps": ", ".join(preprocess_lib.get_maximal_nps(sentence_texts[i])),
      "bert": results_by_review_id[review_id][i]['bert'],
      "dep": results_by_review_id[review_id][i]['dep'],
      })
    with open(f'{args.data_dir}/viewer/{review_id}.html', 'w') as f:
      f.write(html_template.format(build_table_from_rows(rows)))
  with open(f'{args.data_dir}/viewer/index.html', 'w') as f:
      f.write(html_template.format(build_index(text_dict.keys())))
      print(html_template.format(build_index(text_dict.keys())))
  


if __name__ == "__main__":
  main()
