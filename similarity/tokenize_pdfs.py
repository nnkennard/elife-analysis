import glob
import json
import stanza
import tqdm
import xml.etree.ElementTree as ET

import stanza

STANZA_PIPELINE = stanza.Pipeline(
    "en",
    processors="tokenize",
)

PREFIX = "{http://www.tei-c.org/ns/1.0}"
TEXT_ID = f"{PREFIX}text"
BODY_ID = f"{PREFIX}body"
DIV_ID = f"{PREFIX}div"
HEAD_ID = f"{PREFIX}head"
P_ID = f"{PREFIX}p"


def sentence_tokenize(text):
  return [sentence.text for sentence in STANZA_PIPELINE(text).sentences]


def get_docs(filename):
  section_titles = []
  section_texts = []
  divs = (ET.parse(filename).getroot().findall(TEXT_ID)[0].findall(BODY_ID)
          [0].findall(DIV_ID))
  for div in divs:
    (header_node,) = div.findall(HEAD_ID)
    section_titles.append(header_node.text)
    text = ""
    for p in div.findall(P_ID):
      text += " ".join(p.itertext())
    section_texts.append(text)
  return {k: v for k, v in zip(section_titles, section_texts)}


def main():
  tokenized_papers = {}
  for filename in tqdm.tqdm(glob.glob("xmls/*.tei.xml")):
    forum_id = filename[5:-8]
    try:
      section_map = get_docs(filename)
      tokenized_papers[forum_id] = {
          title: sentence_tokenize(text) for title, text in section_map.items()
      }
    except ValueError:
      print(f"Problem with {filename}")

  with open("disapere_tokenized_papers.json", "w") as f:
    json.dump(tokenized_papers, f)


if __name__ == "__main__":
  main()
