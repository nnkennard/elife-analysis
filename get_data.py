import collections
import json
import openreview
import tqdm

def note_is_review(note):
  if not 'review' in note.content:
    return False
  assert len(note.signatures) == 1
  return "AnonReviewer" in note.signatures[0]

Review = collections.namedtuple("Review",
"forum_id review_id text confidence rating".split())

def get_reviews(notes, forum_id):
  reviews = []
  for note in notes:
    if note_is_review(note) and note.replyto == forum_id:
      reviews.append(Review(forum_id, note.id, note.content['review'],
    note.content['confidence'], note.content['rating']))
  return reviews

def main():
  guest_client = openreview.Client(baseurl='https://api.openreview.net')
  offset = 0
  blind_submissions = []
  while True:
    some_blind_submissions = guest_client.get_notes(
        invitation='ICLR.cc/2021/Conference/-/Blind_Submission', offset=offset)
    if not some_blind_submissions:
      break
    blind_submissions += some_blind_submissions
    offset += 1000
  reviews = []
  print("Getting reviews from {0} submissions".format(len(blind_submissions)))
  for sub in tqdm.tqdm(blind_submissions):
    forum_id = sub.forum
    notes = guest_client.get_notes(forum=forum_id)
    reviews += get_reviews(notes, forum_id)

  with open('reviews.jsonl', 'w') as f:
    for review in reviews:
      f.write(json.dumps(review._asdict()) + "\n")



if __name__ == "__main__":
  main()
