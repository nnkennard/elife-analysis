import collections
import joblib
import openreview
import stanza

import elife_lib

FULL_STANZA_PIPELINE = stanza.Pipeline(
    "en",
    processors="tokenize,lemma,pos,depparse,constituency",
)

polarity_classifier = joblib.load("ckpt/polarity_classifier.joblib")

html_template = """
<HTML>
<head>
	<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.3/css/bulma.min.css">
</head>
<body>
<div class="container">
{0}
</div>
</body>
</HTML>
"""


def get_maximal_nps(tree):
  np_spans = []
  stack = [tree]
  while stack:
    this_tree = stack.pop(0)
    if this_tree.label == "NP":
      np_spans.append(" ".join(this_tree.leaf_labels()))
    else:
      stack += this_tree.children
  return np_spans


def build_table_from_rows(rows):
  table_template = """
  <table class="table">
  <thead>
  <tr>
  <th> Index </th> 
  <th> Sentence </th> 
  <th> Polarity label </th> 
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
  </tr>  """.format(row["idx"], row["sentence"], row["polarity"], row["nps"]))
  return table_template.format("\n".join(row_texts))


def get_nps_with_polarity_table(text, split_sentences=True):
  dataframe_rows = []
  doc = FULL_STANZA_PIPELINE(text)
  examples = [
      collections.Counter(elife_lib.extract_dep_paths(sentence))
      for sentence in doc.sentences
  ]
  for i, (sentence, label) in enumerate(
      zip(doc.sentences, polarity_classifier.predict(examples))):
    row_builder = {
        "idx": i,
        "sentence": sentence.text,
        "polarity": label,
    }
    if label == "some_polarity":
      row_builder["nps"] = ", ".join(get_maximal_nps(sentence.constituency))
    else:
      row_builder["nps"] = ""
    dataframe_rows.append(row_builder)

  return build_table_from_rows(dataframe_rows)


def note_is_review(note):
  if not "review" in note.content:
    return False
  assert len(note.signatures) == 1
  return "AnonReviewer" in note.signatures[0]


Review = collections.namedtuple(
    "Review", "forum_id review_id text confidence rating".split())


def get_reviews(sub, client):
  reviews = []
  for note in client.get_notes(forum=sub.forum):
    if note_is_review(note) and note.replyto == sub.forum:
      reviews.append(
          Review(
              sub.forum,
              note.id,
              note.content["review"],
              note.content["confidence"],
              note.content["rating"],
          ))
  return reviews


def main():
  guest_client = openreview.Client(baseurl="https://api.openreview.net")
  offset = 0
  batch_size = 2
  max_reviews = 10
  reviews = []
  while True:
    print(f"Working on submissions {offset} to {offset + batch_size}")
    blind_submissions = guest_client.get_notes(
        invitation="ICLR.cc/2021/Conference/-/Blind_Submission",
        offset=offset,
        limit=batch_size,
    )
    for sub in blind_submissions:
      reviews += get_reviews(sub, guest_client)
    if len(reviews) > max_reviews or not blind_submissions:
      break
    else:
      offset += batch_size
  review_tables = [
      get_nps_with_polarity_table(review.text) for review in reviews
  ]

  with open("polarity_nps_sample.html", "w") as f:
    f.write(html_template.format("<br/><br/>".join(review_tables)))


if __name__ == "__main__":
  main()
