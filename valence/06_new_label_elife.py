import pandas as pd
import os
import glob
import csv
import stanza
import pprint
import texttable
from tqdm import tqdm
from google.cloud import bigquery
from google.cloud import storage
import argparse


pp = pprint.PrettyPrinter(width=100, compact=True)

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="output directory to save labels",
)
parser.add_argument(
    "-a",
    "--annotator",
    type=str,
    help="annotator initials",
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
parser.add_argument(
    "-w",
    "--width",
    type=int,
    default=150,
    help="width of displayed text in chars",
)


parser.add_argument("-ft", "--first_time", action="store_true", help="")
parser.add_argument("-v", "--validate", action="store_true", help="")

# Initialize google clients
BQ_CLIENT = bigquery.Client()
STORAGE_CLIENT = storage.Client()

# Initialize stanza pipeline
SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")

TASKS = "act asp req str pol".split()


mini_labels = {
    "act": "OTH EVL REQ STR FCT HYP SOC".split(),
    "str": "DNU SUM HDG QUO".split(),
    "asp": "OTH MOT ORG SND SUB REP MNG CLR".split(),
    "req": "OTH EDT TYP EXP".split(),
    "pol": "DNU NEG POS".split(),
}

FULL_NAMES = {
    "OTH": "Other",
    "EVL": "act_EVALUATE",
    "REQ": "act_REQUEST",
    "FCT": "act_FACT",
    "STR": "act_STRUCTURING",
    "HYP": "act_HYPOTHESIZE",
    "SOC": "act_SOCIAL",
    "SUM": "struc_SUMMARY",
    "HDG": "struc_HEADING",
    "QUO": "struc_QUOTE",
    "MOT": "asp_MOTIVATION-IMPACT",
    "ORG": "asp_ORIGINALITY",
    "SND": "asp_SOUNDNESS-CORRECTNESS",
    "SUB": "asp_SUBSTANCE",
    "REP": "asp_REPLICABILITY",
    "MNG": "asp_MEANINGFUL-COMPARISON",
    "CLR": "asp_CLARITY",
    "EDT": "req_EDIT",
    "TYP": "req_TYPO",
    "EXP": "req_EXPERIMENT",
    "NEG": "pol_NEGATIVE",
    "POS": "pol_POSITIVE",
}


def make_options_line(options):
    return " | ".join(f"{m}: {i}" for i, m in enumerate(options) if not m == "DNU")


PROMPTS = {
    "act": "Select action: " + make_options_line(mini_labels["act"]),
    "str": "Select structuring type: " + make_options_line(mini_labels["str"]),
    "asp": "Select aspect: " + make_options_line(mini_labels["asp"]),
    "req": "Select request type: " + make_options_line(mini_labels["req"]),
    "pol": "Select polarity: " + make_options_line(mini_labels["pol"]),
}


def make_line(num_chars):
    print("-" * num_chars)


def print_sentence_block(sentences_df, sentence_i, dct, width, mini=False):
    # Print Sentence and its identifiers
    print()
    table_rows = [
        [
            f"SENTENCE {sentence_i + 1} OF {sentences_df.shape[0]} SENTENCES TO RATE IN THIS SESSION"
        ],
        [
            f"M_ID: {dct['manuscript_no']}\tR_ID: {dct['review_id']}\tS_ID: {dct['identifier']}"
        ],
        [f"\n{dct['text']}"],
    ]
    if mini:
        table_rows = [table_rows[2]]
    table_obj = texttable.Texttable(width)
    table_obj.add_rows(table_rows)
    print(table_obj.draw())


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

    # ensure strata correspond with [1,4]
    # < 1 and > 4 occur because bert pretrained on ICLR
    # ensuring strata are in [1,4] also makes
    # groups suffuciently sized to sample within
    df["rating_hat"] = df["rating_hat"].round()
    df["rating_hat"] = df["rating_hat"].replace(5, 4)
    df = df.groupby("rating_hat").sample(n_reviews, random_state=random_seed)

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
            sentence_dct.update(dict.fromkeys(PROMPTS.keys(), int(0)))
            sentence_dicts.append(sentence_dct)
    return pd.DataFrame.from_dict(sentence_dicts)


def get_input(sentence_dct, category):
    print(PROMPTS[category])
    options = mini_labels[category]
    while True:
        try:
            input_val = input()
            int_input = int(input_val)
            if int_input not in range(len(options)):
                print(f"[{input_val}] is not a valid input, please try again.")
            elif int_input == 0 and options[0] == "OTH":
                input_val = input("Write in: ")
                sentence_dct[category] = f"other_{input_val}"
                return
            else:
                sentence_dct[category] = options[int_input]
                return
        except ValueError:
            print(f"[{input_val}] is not a valid input, please try again.")


def print_whole_review(review_id, sentences_df, width):
    table_rows = [["", f"Review: {review_id}"]]
    for i, row in sentences_df[sentences_df["review_id"] == review_id].iterrows():
        table_rows.append([row["identifier"].split("|||")[-1], row["text"]])
    table_obj = texttable.Texttable(width)
    table_obj.set_chars(["", " ", " ", "-"])
    table_obj.add_rows(table_rows)
    print(table_obj.draw())


def label_sentences(whole_sentences_df, n_sents, first_time, file_path, width):
    sentences_df = whole_sentences_df.iloc[:n_sents]

    mode = "w" if first_time else "a"

    with open(file_path, mode=mode) as f:
        writer = csv.DictWriter(f, sentences_df.columns)
        if first_time:
            writer.writeheader()

        for i, (_, pre_sentence_dct) in enumerate(sentences_df.iterrows()):
            redo = True
            while redo:
                if pre_sentence_dct["identifier"].endswith("|||0"):
                    print_whole_review(
                        pre_sentence_dct["review_id"], whole_sentences_df, width
                    )

                sentence_dct = pre_sentence_dct.to_dict()
                print_sentence_block(sentences_df, i, sentence_dct, width)

                get_input(sentence_dct, "act")
                if sentence_dct["act"] == "REQ":
                    get_input(sentence_dct, "req")
                    get_input(sentence_dct, "asp")
                    sentence_dct["pol"] = "NEG"

                elif sentence_dct["act"] == "STR":
                    get_input(sentence_dct, "str")

                elif sentence_dct["act"] == "EVL":
                    get_input(sentence_dct, "pol")
                    get_input(sentence_dct, "asp")

                rows = [["Task", "Label"]]
                for t in TASKS:
                    val = sentence_dct[t]
                    if val:
                        rows.append(
                            [t, FULL_NAMES.get(sentence_dct[t], sentence_dct[t])]
                        )
                print("\n\nLabels for this sentence:")
                table_obj = texttable.Texttable(width)
                table_obj.set_chars([" ", " ", " ", "-"]).set_header_align(["l"])
                table_obj.add_rows([[sentence_dct["text"]]])
                print(table_obj.draw())
                # print("\n"+sentence_dct['text'])
                table_obj = texttable.Texttable(width)
                table_obj.set_chars(["", " ", " ", "-"])
                table_obj.add_rows(rows)
                print(table_obj.draw())

                valid_redo_input = False
                while not valid_redo_input:
                    input_val = input("Would you like to redo? (y/n): ")
                    if input_val in "nN":
                        redo = False
                        valid_redo_input = True
                    elif input_val not in "yY":
                        valid_redo_input = False
                    else:
                        valid_redo_input = True
                        redo = True

            writer.writerow(sentence_dct)


def hello():
    print(
        "\n".join(
            [
                "\n" * 2,
                "+" * 33,
                "START INTERACTIVE SESSION!",
                "+" * 33,
            ]
        )
    )


def goodbye():
    print()
    print("+" * 33)
    print("END INTERACTIVE SESSION!")
    print("+" * 33)
    print("\n" * 3)


def get_file_path(args):
    return f"{args.output_dir}/values_annotation_{args.annotator}_rs_{args.random_seed}_nr_{args.n_reviews}.csv"


def main():

    args = parser.parse_args()

    file_path = get_file_path(args)
    # Get data
    sentences_df = summon_reviews(args.n_reviews, args.random_seed)
    sentences_df = get_sentences_df(sentences_df)

    # Begin
    hello()

    # if labeling:

    if args.validate:
        flags = input("Enter sentence ids: ").split(",")
        for flag in flags:
            table_obj = texttable.Texttable(args.width)
            table_obj.set_chars([" ", " ", " ", "-"]).set_header_align(["l"])
            table_obj.add_rows(
                [[sentences_df[sentences_df["identifier"] == flag]["text"].iloc[0]]]
            )
            print(table_obj.draw())
            advance = eval(input("Advance?: "))

    else:

        # first time means new file
        if args.first_time:
            print(f"{sentences_df.shape[0]} total sentences to label.")
            label_sentences(
                sentences_df, args.n_sents, args.first_time, file_path, args.width
            )

        # nth time means append file, and make sure only unrated reviews are printed
        else:

            # open existing rated reviews file
            rater_df = pd.read_csv(file_path)
            already_reviewed = list(rater_df["identifier"])
            truncated_sentences_df = sentences_df[
                ~sentences_df["identifier"].isin(already_reviewed)
            ]
            print(f"{truncated_sentences_df.shape[0]} sentences left to label.")
            if not truncated_sentences_df.iloc[0]["identifier"].endswith("|||0"):
                print_whole_review(
                    truncated_sentences_df.iloc[0]["review_id"],
                    sentences_df,
                    args.width,
                )
            label_sentences(
                truncated_sentences_df,
                args.n_sents,
                args.first_time,
                file_path,
                args.width,
            )

    # End
    goodbye()


if __name__ == "__main__":
    main()
