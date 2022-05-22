import numpy as np
from sentence_transformers import SentenceTransformer
import stanza

STANZA_PIPELINE = stanza.Pipeline("en",
                                  processors="tokenize",
                                  tokenize_no_ssplit=True)




SBERT_MODEL = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def cosine(vec_a, vec_b):
  return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def sbert(sentences_1, sentences_2):
  results = np.zeros([len(sentences_1), len(sentences_2)])
  embeddings_1 = SBERT_MODEL.encode(sentences_1)
  embeddings_2 = SBERT_MODEL.encode(sentences_2)
  for i, e1 in enumerate(embeddings_1):
    for j, e2 in enumerate(embeddings_2):
      results[i][j] = cosine(e1, e2)


def jaccard_tokenize(sentences):
  processed = STANZA_PIPELINE("\n\n".join(sentences))
  return [[token.text
           for token in sentence.tokens]
          for sentence in processed.sentences]

def jaccard_similarity(s1, s2):
  return len(set(s1).intersection(set(s2))) / len(set(s1).union(set(s2)))

def jaccard(sentences_1, sentences_2):
  results = np.zeros([len(sentences_1), len(sentences_2)])
  tokens_1 = jaccard_tokenize(sentences_1)
  tokens_2 = jaccard_tokenize(sentences_2)
  for i, t1 in enumerate(tokens_1):
    for j, t2 in enumerate(tokens_2):
      results[i][j] = jaccard_similarity(t1,t2)
  return results


FUNCTION_MAP = {
  "sbert": sbert,
  "jaccard": jaccard
}

