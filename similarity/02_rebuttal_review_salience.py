import argparse
import collections
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import stanza
import tqdm

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

STANZA_PIPELINE = stanza.Pipeline("en",
                                  processors="tokenize",
                                  tokenize_no_ssplit=True)


def cosine(vec_a, vec_b):
  return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def calculate_sbert_similarities(example, model):
  review_embeddings = model.encode(example["tokenized_review"])
  rebuttal_embeddings = model.encode(example["tokenized_rebuttal"])
  results = np.zeros([len(rebuttal_embeddings), len(review_embeddings)])
  for reb_i, reb_embedding in enumerate(rebuttal_embeddings):
    for rev_i, rev_embedding in enumerate(review_embeddings):
      results[reb_i][rev_i] = cosine(reb_embedding, rev_embedding)
  return results


def jaccard_tokenize(sentences):
  processed = STANZA_PIPELINE("\n\n".join(sentences))
  return [[token.text
           for token in sentence.tokens]
          for sentence in processed.sentences]


def jaccard(s1, s2):
  return len(set(s1).intersection(set(s2))) / len(set(s1).union(set(s2)))


def calculate_jaccard_similarities(example):
  review_token_lists = jaccard_tokenize(example["tokenized_review"])
  rebuttal_token_lists = jaccard_tokenize(example["tokenized_rebuttal"])
  results = np.zeros([len(rebuttal_token_lists), len(review_token_lists)])
  for reb_i, reb_tokens in enumerate(rebuttal_token_lists):
    for rev_i, rev_tokens in enumerate(review_token_lists):
      results[reb_i][rev_i] = jaccard(reb_tokens, rev_tokens)
  return results


def calculate_rebuttal_review_salience(example, sbert_model):
  return {
      "sbert": calculate_sbert_similarities(example, sbert_model).tolist(),
      "jaccard": calculate_jaccard_similarities(example).tolist(),
  }


def main():

  args = parser.parse_args()

  sbert_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

  with open(args.input_file, "r") as f:
    input_data = json.load(f)

  results = []
  for structure in tqdm.tqdm(input_data["structures"]):
    for review in structure["reviews"]:
      result = calculate_rebuttal_review_salience(review, sbert_model)
      results.append({
          "review_id": review["review_id"],
          "review_sentences": review["tokenized_review"],
          "rebuttal_sentences": review["tokenized_rebuttal"],
          "result": result,
      })

  with open(args.output_file, "w") as f:
    json.dump(results, f)


if __name__ == "__main__":
  main()
