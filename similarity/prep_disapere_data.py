import collections
import glob
import json
import openreview
import numpy as np
import tqdm

DISAPERE_ENV = "../data/DISAPERE/final_dataset/"
SUBSETS = "train dev test".split()

Example = collections.namedtuple("Example",
  "review_sentences rebuttal_sentences alignment forum_id")

GUEST_CLIENT = openreview.Client(baseurl='https://api.openreview.net')

def process_example(filename):
  with open(filename, 'r') as f:
    obj = json.load(f)
    review_sentences = [sentence["text"] for sentence in obj["review_sentences"]]
    rebuttal_sentences = [sentence["text"] for sentence in obj["rebuttal_sentences"]]
    alignment = np.zeros([len(rebuttal_sentences), len(review_sentences)])
    for reb_i, sentence in enumerate(obj['rebuttal_sentences']):
      align_type, indices = sentence['alignment']
      if indices is not None:
        for rev_i in indices:
          alignment[reb_i][rev_i] = 1

  return Example(
    review_sentences, rebuttal_sentences, alignment.tolist(),
    obj['metadata']['forum_id']
  )

def write_pdf(forum_id, pdf_directory):
  pdf_binary = GUEST_CLIENT.get_pdf(forum_id, is_reference=False)
  with open(f'{pdf_directory}/{forum_id}.pdf', 'wb') as file_handle:
    file_handle.write(pdf_binary)

def main():
  written_forums = []
  examples = collections.defaultdict(list)
  for subset in SUBSETS:
    print(subset)
    for filename in tqdm.tqdm(glob.glob(f'{DISAPERE_ENV}/{subset}/*')):
      example = process_example(filename)
      examples[subset].append(example._asdict())
      if example.forum_id not in written_forums:
        write_pdf(example.forum_id, 'pdfs/')
        written_forums.append(example.forum_id)

  with open('disapere_intermediate.json', 'w') as f:
    json.dump({"pdf_dir":"pdfs/", "subsets":examples}, f)

if __name__ == "__main__":
  main()
