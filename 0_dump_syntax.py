import collections
import glob
import json
import pickle
import stanza
import tqdm

STANZA_PIPELINE = stanza.Pipeline('en',
                                  processors='tokenize,lemma,pos,depparse',
                                  tokenize_no_ssplit=True)
no_polarity = []
some_polarity = []
featurized_by_polarity = collections.defaultdict(list)


def featurize(pre_doc):
  annotated = STANZA_PIPELINE(pre_doc)
  return annotated.sentences


for split in "train dev test".split():
  for filename in tqdm.tqdm(
      glob.glob(f"../../DISAPERE/DISAPERE/final_dataset/{split}/*")):
    with open(filename, 'r') as f:
      obj = json.load(f)
      review_doc = "\n\n".join([x['text'] for x in obj["review_sentences"]])
      featurized_sentences = featurize(review_doc)
      for orig_sentence, featurized_sentence in zip(obj["review_sentences"],
                                                    featurized_sentences):
        if orig_sentence['pol'] == 'none':
          featurized_by_polarity['no_polarity'].append(featurized_sentence)
        else:
          featurized_by_polarity['some_polarity'].append(featurized_sentence)

  with open(f"featurized_by_polarity_{split}.pkl", 'wb') as f:
    pickle.dump(featurized_by_polarity, f)
