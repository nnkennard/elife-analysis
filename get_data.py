import collections
import json
import openreview
import os
import stanza
import tqdm

STANZA_PIPELINE = stanza.Pipeline('en',
                                  processors='tokenize,lemma,pos,depparse')


def note_is_review(note):
  if not 'review' in note.content:
    return False
  assert len(note.signatures) == 1
  return "AnonReviewer" in note.signatures[0]


ReviewMetadata = collections.namedtuple(
    "ReviewMetadata", "forum_id review_id confidence rating".split())

STANZA_KEYS = "text xpos head deprel".split()


class Review(object):

  def __init__(self, forum_id, review_id, text, confidence, rating):
    self.metadata = ReviewMetadata(forum_id, review_id, confidence,
                                   rating)._asdict()
    self.text = text
    self.syntax = self._get_syntax(text)

  def _get_syntax(self, text):
    doc = STANZA_PIPELINE(text)
    sentences = []
    for sentence in doc.sentences:
      sentences.append(
          [self._get_token_info(token) for token in sentence.tokens])
    syntax = {}
    for key in STANZA_KEYS:
      syntax[key] = [[tok[key] for tok in sentence] for sentence in sentences]
    return syntax

  def _get_token_info(self, token):
    token_dict, = token.to_dict()
    return {key: token_dict[key] for key in STANZA_KEYS}

  def as_json(self):
    dict_prep = {"original_text" : self.text}
    dict_prep.update(self.metadata)
    dict_prep.update(self.syntax)
    return json.dumps(dict_prep)


def get_reviews(notes, forum_id):
  reviews = []
  for note in notes:
    if note_is_review(note) and note.replyto == forum_id:
      reviews.append(
          Review(forum_id, note.id, note.content['review'],
                 note.content['confidence'], note.content['rating']))

  return reviews


def save_reviews(forum_id, client, data_dir):
  notes = client.get_notes(forum=forum_id)
  for review in get_reviews(notes, forum_id):
    review_dir = data_dir + "/" + forum_id
    os.makedirs(review_dir, exist_ok=True)
    with open(review_dir + "/" + review.metadata['review_id'] + ".json", 'w') as f:
      f.write(review.as_json())


def main():
  guest_client = openreview.Client(baseurl='https://api.openreview.net')
  data_dir = "reviews/"
  offset = 0
  limit = 100
  while True:
    print("Working on submissions {0} to {1}".format(offset, offset + limit))
    blind_submissions = guest_client.get_notes(
        invitation='ICLR.cc/2021/Conference/-/Blind_Submission',
        offset=offset,
        limit=limit)
    if blind_submissions:
      offset += limit
    else:
      break
    for sub in tqdm.tqdm(blind_submissions):
      save_reviews(sub.forum, guest_client, data_dir)


if __name__ == "__main__":
  main()
