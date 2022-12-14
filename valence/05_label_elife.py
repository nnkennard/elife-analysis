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
    "-n",
    "--n_reviews",
    type=str,
    help="n reviews to label by hand",
)

# Initialize google clients
BQ_CLIENT = bigquery.Client()
STORAGE_CLIENT = storage.Client()

# Initialize stanza pipeline
SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")

# DISAPERE Labels
ARGS = [
    "REQUEST",
    "FACT",
    "STRUCTURING",
    "SOCIAL",
]

ASPS = [
    "MOTIVATION-IMPACT",
    "ORIGINALITY",
    "SOUNDNESS-CORRECTNESS",
    "SUBSTANCE",
    "REPLICABILITY",
    "MEANINGFUL-COMPARISON",
    "CLARITY",
]

PATH = "/home/jupyter/00_daniel/00_reviews/00_data/"


def GetInfo():
    """
    Requests conder info to use in writing/appending
    the their own labeled reviews csv
    """
    first_time = input("This is your first time (True/False): ").capitalize()
    rater = input("Last name: ").lower()
    n_to_rate = input(
        "How many review sentences do you want to rate?\n(Fewer means more breaks): "
    )
    return eval(first_time), rater, eval(n_to_rate)


def summon_reviews(n_reviews):
    """
    Returns a pandas DF containing n sampled eLife reviews
    by summoning reviews from Google BQ.
    """

    REVIEW_QRY = """
    SELECT Manuscript_no_, review_id, rating_hat, Major_comments,
    FROM `gse-nero-dmcfarla-mimir`.eLife.eLife_Reviews_IDRating
    """

    df = BQ_CLIENT.query(REVIEW_QRY).result().to_dataframe()
    df_nonans = df.dropna()
    return df_nonans.sample(n_reviews, random_state=72)


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
            sentence_dicts.append(
                {
                    "manuscript_no": ms_id,
                    "review_id": review_id,
                    "identifier": _make_identifier(review_id, i),
                    "text": sentence.text,
                }
            )
    return pd.DataFrame.from_dict(sentence_dicts)


def main(n_reviews):

    # Get data
    print("Getting data from BQ...")
    df = summon_reviews(n_reviews)
    sentence_df = get_sentences_df(df)

    # Get user info
    print()
    print()
    print()
    print("*" * 33)
    print("-" * 33)
    print("START INTERACTIVE CODING SESSION!")
    print("-" * 33)
    print("*" * 33)
    first_time, rater, n_to_rate = GetInfo()
    rater_file = "{}_disapere_elife_labels.csv".format(rater)

    # first time means new file
    if first_time == True:

        sentence_df = sentence_df.iloc[:n_to_rate]
        n_reviews = 0
        for _, review_df in sentence_df.groupby("review_id"):
            n_reviews += 1
            n_sentences = 0

            sentence_dicts = []
            for _, sentence_dct in review_df.iterrows():
                n_sentences += 1
                mid = sentence_dct["manuscript_no"]
                rid = sentence_dct["review_id"]
                sid = sentence_dct["identifier"]
                print()
                print("-" * 100)
                print(
                    f"SENTENCE {n_sentences} OF {len(review_df)} SENTENCES IN REVIEW {n_reviews} of {n_to_rate} REVIEWS"
                )
                print(f"M_ID: {mid}\tR_ID: {rid}\tS_ID: {sid}")
                print("-" * 50)
                pp.pprint(f"{sentence_dct['text']}")
                print("-" * 100)

                evaluative = input(
                    "\tThis sentence subjectively evaluates the manuscript (0=no, 1=yes): "
                )
                sentence_dct["arg_evaluative"] = int(evaluative)

                if sentence_dct["arg_evaluative"] == 0:

                    print("\n\tSelect the non-evaluative action of this sentence:")
                    for arg in ARGS:
                        key = f"arg_{arg.lower()}"
                        value = input(f"\t\t{arg}: ")
                        sentence_dct[key] = int(value)

                    if sentence_dct["arg_request"] == 1:

                        print("\n\tSelect what this sentence requests:")
                        for req in "Experiment Edit Typo".split():
                            key = f"req_{req.lower()}"
                            value = input(f"\t\t{req}: ")
                            sentence_dct[key] = int(value)

                        print(
                            "\n\tSelect the aspect of the manuscript that is the subject of this sentence's request:"
                        )
                        for asp in ASPS:
                            key = f"asp_{asp.lower()}"
                            value = input(f"\t\t{asp}: ")
                            sentence_dct[key] = int(value)

                    else:
                        # Null vals for asps when arg is non-eval
                        for asp in ASPS:
                            key = f"asp_{asp.lower()}"
                            sentence_dct[key] = 0

                else:

                    # Null vals for other args when arg is eval
                    for arg in ARGS:
                        key = f"arg_{arg.lower()}"
                        sentence_dct[key] = 0

                    print(
                        "\n\tSelect the aspect of the manuscript that this sentence evaluates:"
                    )
                    for asp in ASPS:
                        key = f"asp_{asp.lower()}"
                        value = input(f"\t\t{asp}: ")
                        sentence_dct[key] = int(value)
                sentence_dicts.append(sentence_dct)

        sentences_df = pd.DataFrame.from_dict(sentence_dicts)
        sentences_df.to_csv(PATH + rater_file, header=True, mode="w", index=False)

    # nth time means append file, and make sure only unrated reviews are printed
    if first_time == False:

        # open existing rated reviews file
        rater_df = pd.read_csv(PATH + rater_file)
        already_reviewed = list(rater_df["identifier"])

        n_reviews = 0
        sentence_df = sentence_df[~sentence_df["identifier"].isin(already_reviewed)][
            :n_to_rate
        ]
        for review_id, review_df in sentence_df.groupby("review_id"):
            n_reviews += 1
            n_sentences = 0
            sentence_dicts = []
            for _, sentence_dct in review_df.iterrows():
                n_sentences += 1
                mid = sentence_dct["manuscript_no"]
                rid = sentence_dct["review_id"]
                sid = sentence_dct["identifier"]
                print()
                print("-" * 100)
                print(
                    f"SENTENCE {n_sentences} OF {len(review_df)} SENTENCES IN REVIEW {n_reviews} of {n_to_rate} REVIEWS"
                )
                print(f"M_ID: {mid}\tR_ID: {rid}\tS_ID: {sid}")
                print("-" * 50)
                pp.pprint(f"{sentence_dct['text']}")
                print("-" * 100)

                evaluative = input(
                    "\tThis sentence subjectively evaluates the manuscript (0=no, 1=yes): "
                )
                sentence_dct["arg_evaluative"] = int(evaluative)

                if sentence_dct["arg_evaluative"] == 0:

                    print("\n\tSelect the non-evaluative action of this sentence:")
                    for arg in ARGS:
                        key = f"arg_{arg.lower()}"
                        value = input(f"\t\t{arg}: ")
                        sentence_dct[key] = int(value)

                    if sentence_dct["arg_request"] == 1:

                        print("\n\tSelect what this sentence requests:")
                        for req in "Experiment Edit Typo".split():
                            key = f"req_{req.lower()}"
                            value = input(f"\t\t{req}: ")
                            sentence_dct[key] = int(value)

                        print(
                            "\n\tSelect the aspect of the manuscript that is the subject of this sentence's request:"
                        )
                        for asp in ASPS:
                            key = f"asp_{asp.lower()}"
                            value = input(f"\t\t{asp}: ")
                            sentence_dct[key] = int(value)

                    else:
                        # Null vals for asps when arg is non-eval
                        for asp in ASPS:
                            key = f"asp_{asp.lower()}"
                            sentence_dct[key] = 0

                else:

                    # Null vals for other args when arg is eval
                    for arg in ARGS:
                        key = f"arg_{arg.lower()}"
                        sentence_dct[key] = 0

                print(
                    "\n\tSelect the aspect of the manuscript that this sentence evaluates:"
                )
                for asp in ASPS:
                    key = f"asp_{asp.lower()}"
                    value = input(f"\t\t{asp}: ")
                    sentence_dct[key] = int(value)

                sentence_dicts.append(sentence_dct)

            sentences_df = pd.DataFrame.from_dict(sentence_dicts)
            sentences_df.to_csv(PATH + rater_file, header=False, mode="a", index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(int(args.n_reviews))
