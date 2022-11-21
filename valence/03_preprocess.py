import argparse
import glob
import json
import os
import stanza
import tqdm

import classification_lib

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-r",
    "--raw_data_dir",
    type=str,
    help="path to data file containing raw review jsons",
)
parser.add_argument(
    "-p",
    "--processed_data_file",
    type=str,
    help="path to data file containing score jsons",
)

SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")



def get_sentences_from_file(filename):
    sentences = []
    with open(filename, "r") as f:
        obj = json.load(f)
        for i, sentence in enumerate(
            SENTENCIZE_PIPELINE(obj["text"]).sentences
        ):
            sentences.append(
                {
                    "identifier": classification_lib.make_identifier(obj["identifier"], i),
                    "text": sentence.text,
                }
            )
    return sentences


def main():

    args = parser.parse_args()

    with open(args.processed_data_file, "w") as f:
        for filename in glob.glob(f"{args.raw_data_dir}/*.json"):
            for sentence in get_sentences_from_file(filename):
                f.write(json.dumps(sentence) + "\n")


if __name__ == "__main__":
    main()
