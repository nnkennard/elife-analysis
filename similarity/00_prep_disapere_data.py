import argparse
import collections
import glob
import json
import openreview
import numpy as np
import tqdm

parser = argparse.ArgumentParser(
    description="Prepare DISAPERE data for similarity calculations")
parser.add_argument(
    "-d",
    "--disapere_path",
    default="../data/DISAPERE/final_dataset/",
    type=str,
    help="path to DISAPERE final_dataset directory",
)
parser.add_argument(
    "-p",
    "--pdf_dir",
    default="pdfs/",
    type=str,
    help="path to directory where pdfs will be saved",
)
parser.add_argument(
    "-o",
    "--output_file",
    default="disapere_similarity_input.json",
    type=str,
    help="output json filename",
)

GUEST_CLIENT = openreview.Client(baseurl="https://api.openreview.net")
Example = collections.namedtuple(
    "Example",
    "review_text reviewer_id rebuttal_text alignment forum_id review_id")


def process_example(filename):
  with open(filename, "r") as f:
    obj = json.load(f)
    review_sentences = [
        sentence["text"] for sentence in obj["review_sentences"]
    ]
    rebuttal_sentences = [
        sentence["text"] for sentence in obj["rebuttal_sentences"]
    ]
    alignment = np.zeros([len(rebuttal_sentences), len(review_sentences)])
    for reb_i, sentence in enumerate(obj["rebuttal_sentences"]):
      align_type, indices = sentence["alignment"]
      if indices is not None:
        for rev_i in indices:
          alignment[reb_i][rev_i] = 1

  return Example(
      "\n\n".join(review_sentences),
      obj["metadata"]["reviewer"],
      "\n\n".join(rebuttal_sentences),
      alignment.tolist(),
      obj["metadata"]["forum_id"],
      obj["metadata"]["review_id"],
  )


def write_pdf(forum_id, pdf_directory):
  pdf_binary = GUEST_CLIENT.get_pdf(forum_id, is_reference=False)
  pdf_path = f"{pdf_directory}/{forum_id}.pdf"
  with open(pdf_path, "wb") as file_handle:
    file_handle.write(pdf_binary)
  return {
      "forum_id": forum_id,
      "manuscript_pdf_path": pdf_path,
  }


def build_structures(examples):
  forum_to_reviews = collections.defaultdict(list)
  review_map = {}
  for example in examples:
    forum_to_reviews[example.forum_id].append(example.review_id)
    review_map[example.review_id] = example

  structure_list = []
  for forum, reviews in forum_to_reviews.items():
    structure_builder = {"forum_id": forum, "reviews": []}
    for review_id in reviews:
      review = review_map[review_id]
      assert review_id == review.review_id
      structure_builder["reviews"].append({
          "review_id": review.review_id,
          "review_text": review.review_text,
          "reviewer_id": review.reviewer_id,
          "rebuttal_text": review.rebuttal_text,
      })
    structure_list.append(structure_builder)
  return structure_list


def main():

  args = parser.parse_args()

  manuscript_list = []
  pdfs_downloaded = []
  examples = []
  for filename in tqdm.tqdm(glob.glob(f"{args.disapere_path}/train/*")):
    example = process_example(filename)
    examples.append(example)
    if example.forum_id not in pdfs_downloaded:
      manuscript_list_item = write_pdf(example.forum_id, args.pdf_dir)
      pdfs_downloaded.append(example.forum_id)
      manuscript_list.append(manuscript_list_item)

  structures = build_structures(examples)

  with open(args.output_file, "w") as f:
    json.dump(
        {
            "pdf_dir": args.pdf_dir,
            "structures": build_structures(examples),
            "manuscript_files": manuscript_list,
        },
        f,
    )


if __name__ == "__main__":
  main()
