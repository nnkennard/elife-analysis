import pandas as pd
import os
import glob
import csv
import stanza
import pprint
from tqdm import tqdm
from google.cloud import bigquery
from google.cloud import storage
import argparse


pp = pprint.PrettyPrinter(width=100, compact=True)

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-fp",
    "--file_path",
    type=str,
    help="full file path to save labels",
)
parser.add_argument(
    "-nr",
    "--n_reviews",
    type=int,
    help="n reviews to randomly sample",
)
parser.add_argument(
    "-ns",
    "--n_sents",
    type=int,
    help="n sentences to label by hand",
)
parser.add_argument(
    "-rs",
    "--random_seed",
    type=int,
    help="random seed",
)


parser.add_argument("-ft", "--first_time", action="store_true", help="")
parser.add_argument("-v", "--validate", action="store_true", help="")

# Initialize google clients
BQ_CLIENT = bigquery.Client()
STORAGE_CLIENT = storage.Client()

# Initialize stanza pipeline
SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")

# DISAPERE Labels
ARGS = [
    "arg_OTHER",
    "arg_EVALUATIVE",
    "arg_REQUEST",
    "arg_FACT",
    "arg_STRUCTURING",
    "arg_SOCIAL",
]

ASPS = [
    "asp_MOTIVATION-IMPACT",
    "asp_ORIGINALITY",
    "asp_SOUNDNESS-CORRECTNESS",
    "asp_SUBSTANCE",
    "asp_REPLICABILITY",
    "asp_MEANINGFUL-COMPARISON",
    "asp_CLARITY",
    "asp_OTHER",
]

REQS = ["req_EDIT", "req_TYPO", "req_EXPERIMENT"]

STRS = ["struc_SUMMARY", "struc_HEADING", "struc_QUOTE"]

ALL = ARGS + ASPS + REQS + STRS

ALL.extend(["neg_polarity", "pos_polarity"])

TASKS = "act asp req str pol".split()

PROMPTS = {
    "act": "\n\tSelect the action of this sentence:",
    "req": "\n\tSelect what this sentence requests:",
    "req_asp": "\n\tSelect the aspect of the manuscript that is the subject of this sentence's request:",
    "struct": "\n\tSelect the kind of structuring of this sentence:",
    "pol": "\n\tIs the evaluation positive? (0/1): ",
    "evl_asp": "\n\tSelect the aspect of the manuscript that this sentence evaluates:",
}


def make_line(num_chars):
    print("-" * num_chars)


def print_sentence_block(sentences_df, sentence_i, dct):
    # Print Sentence and its identifiers
    print()
    make_line(100)
    print(
        f"SENTENCE {sentence_i + 1} OF {sentences_df.shape[0]} SENTENCES TO RATE IN THIS SESSION"
    )
    print(
        f"M_ID: {dct['manuscript_no']}\tR_ID: {dct['review_id']}\tS_ID: {dct['identifier']}"
    )
    make_line(50)
    pp.pprint(f"{dct['text']}")
    make_line(100)


def summon_reviews(n_reviews, random_seed=7272):
    """
    Returns a pandas DF containing n sampled eLife reviews
    by summoning reviews from Google BQ.
    """

    REVIEW_QRY = """
    SELECT Manuscript_no_, review_id, rating_hat, Major_comments,
    FROM `gse-nero-dmcfarla-mimir`.eLife.eLife_Reviews_IDRating
    """
    print("Getting data from BQ...")
    df = BQ_CLIENT.query(REVIEW_QRY).result().to_dataframe()
    df = df.dropna()
    # TODO: sample within score strata
    return df.sample(n_reviews, random_state=random_seed)


def _make_identifier(review_id, index):
    return f"{review_id}|||{index}"


def get_sentences_df(df):
    """
    Tokenizes review sentence and
    Returns a df where row is sentence.
    """
    sentence_dicts = []
    for i, row in tqdm(df.iterrows()):
        review_id = row["review_id"]
        raw_text = row["Major_comments"]
        ms_id = row["Manuscript_no_"]
        for i, sentence in enumerate(SENTENCIZE_PIPELINE(raw_text).sentences):
            sentence_dct = {
                "manuscript_no": ms_id,
                "review_id": review_id,
                "identifier": _make_identifier(review_id, i),
                "text": sentence.text,
            }
            sentence_dct.update(dict.fromkeys([all.lower() for all in ALL], int(0)))
            sentence_dicts.append(sentence_dct)
    return pd.DataFrame.from_dict(sentence_dicts)


def get_input(label_name):
    if 'OTHER' in label_name:
      return input(f"\t\t{label_name} (write it in): ")
    else:
      return int(input(f"\t\t{label_name}: "))


def label_sentences(sentences_df, n_sents, first_time, file_path):
    sentences_df = sentences_df.iloc[:n_sents]

    mode = "w" if first_time else "a"

    with open(file_path, mode=mode) as f:
        writer = csv.DictWriter(f, sentences_df.columns)
        if first_time:
            writer.writeheader()

        for i, sentence_dct in sentences_df.iterrows():
            sentence_dct = sentence_dct.to_dict()
            print_sentence_block(sentences_df, i, sentence_dct)

            print(PROMPTS["act"])
            for arg in ARGS:
                sentence_dct[arg.lower()] = get_input(arg)

            if sentence_dct["arg_request"] == 1:
                print(PROMPTS["req"])
                for req in REQS:
                    sentence_dct[req.lower()] = get_input(req)

                print(PROMPTS["req_asp"])
                for asp in ASPS:
                  sentence_dct[asp.lower()] = get_input(asp)

                sentence_dct["neg_polarity"] = 1

            elif sentence_dct["arg_structuring"] == 1:
                print(PROMPTS['struct'])
                for struc in STRS:
                    sentence_dct[struc.lower()] = get_input(struc)

            elif sentence_dct["arg_evaluative"] == 1:
                print(PROMPTS['pol'])
                sentence_dct["pos_polarity"] = get_input('pol')
                sentence_dct["neg_polarity"] = 1- sentence_dct['pos_polarity']

                print(PROMPTS['evl_asp'])
                for asp in ASPS:
                    sentence_dct[asp.lower()] = get_input(asp)

            writer.writerow(sentence_dct)


def hello():
    print("\n" * 3)
    print("+" * 33)
    print("START INTERACTIVE SESSION!")
    print("+" * 33)


def goodbye():
    print()
    print("+" * 33)
    print("END INTERACTIVE SESSION!")
    print("+" * 33)
    print("\n" * 3)


def main():

    args = parser.parse_args()
    # Get data
    sentences_df = summon_reviews(args.n_reviews, args.random_seed)
    sentences_df = get_sentences_df(sentences_df)

    # Begin
    hello()

    # if labeling:

    if args.validate:
        flags = input("Enter sentence ids: ").split(",")
        for flag in flags:
            sent = sentences_df[sentences_df["identifier"] == flag]["text"].iloc[0]
            pp.pprint(sent)
            print()
            advance = eval(input("Advance?: "))

    else:

        # first time means new file
        if args.first_time:
            print(f"{sentences_df.shape[0]} total sentences to label.")
            label_sentences(sentences_df, args.n_sents, args.first_time, args.file_path)

        # nth time means append file, and make sure only unrated reviews are printed
        else:
            # open existing rated reviews file
            rater_df = pd.read_csv(file_path)
            already_reviewed = list(rater_df["identifier"])
            sentences_df = sentences_df[
                ~sentences_df["identifier"].isin(already_reviewed)
            ]
            print(f"{sentences_df.shape[0]} sentences left to label.")
            label_sentences(sentences_df, args.n_sents, args.first_time, args.file_path)

    # End
    goodbye()


if __name__ == "__main__":
    main()
