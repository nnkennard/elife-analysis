import argparse
import glob
import json
import stanza
import tqdm
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description="Prepare tokenized paper text")
parser.add_argument(
    "-i",
    "--input_file",
    default="disapere_similarity_input.json",
    type=str,
    help="input json file",
)
parser.add_argument(
    "-x",
    "--xml_dir",
    default="xmls/",
    type=str,
    help="path to xmls as parsed by grobid",
)
parser.add_argument(
    "-o",
    "--output_dir",
    default="disapere_results/",
    type=str,
    help="output json filename",
)

STANZA_PIPELINE = stanza.Pipeline("en",
                                  processors="tokenize",
                                  tokenize_no_ssplit=True)

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

  args = parser.parse_args()

  with open(args.input_file, "r") as f:
    input_obj = json.load(f)

  forums_to_get = [x["forum_id"] for x in input_obj["manuscript_files"]]

  # Tokenize reviews and rebuttals
  for structure in input_obj["structures"]:
    for review in structure["reviews"]:
      review["review_sentences"] = sentence_tokenize(review["review_text"])
      if review["rebuttal_text"] is not None:
        review["rebuttal_sentences"] = sentence_tokenize(
            review["rebuttal_text"])

  # Tokenize manuscripts
  input_obj["manuscript_sentences"] = []
  for filename in tqdm.tqdm(glob.glob(f"{args.xml_dir}/*.tei.xml")):
    forum_id = filename[5:-8]
    if forum_id not in forums_to_get:
      continue
    try:
      section_map = get_docs(filename)
      input_obj["manuscript_sentences"].append({
          "forum_id": forum_id,
          "manuscript_sentences": {
              title: sentence_tokenize(text)
              for title, text in section_map.items()
          },
      })
    except ValueError:
      print(f"Problem with {filename}")

  with open(f'{args.output_file}/similarity_tokenized.json', "w") as f:
    json.dump(input_obj, f)


if __name__ == "__main__":
  main()
