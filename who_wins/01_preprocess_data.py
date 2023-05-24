import argparse
import collections
import glob
import gzip

from interval import Interval
import json
import os
import stanza
import pandas as pd
from tqdm import tqdm

# from google.cloud import bigquery
import iclr_lib

parser = argparse.ArgumentParser(description="")
parser.add_argument("-d", "--data_dir", default="", type=str, help="")


SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")
TOLERANCE = 7
Sentence = collections.namedtuple("Sentence", "interval text")


def tokenize(text):
    doc = SENTENCIZE_PIPELINE(text)
    sentences = []
    for sentence in doc.sentences:
        start = sentence.to_dict()[0]["start_char"]
        end = sentence.to_dict()[-1]["end_char"]
        sentences.append(Sentence(Interval(start, end), sentence.text))
    return sentences


# ==== eLife
# map old elife hand labels
# onto new eLife labels
elife_pol_map = {"0": "non", "POS": "pos", "NEG": "neg"}
# elife_asp_map = {
#     "0": "non",
#     "SND": "snd",
#     "ORG": "org",
#     "MOT": "mot",
#     "SUB": "sbs",
#     "MNG": "mng",
#     "CLR": "clr",
#     "REP": "rep",
#     "other_ms": "non",
#     "other_0": "non",
#     "other_": "non",
#     "other_n": "non",
#     "other_manuscript": "non",
# }
elife_asp_map = {
    "0": "non",
    "SND": "acc",
    "ORG": "nvl",
    "MOT": "nvl",
    "SUB": "acc",
    "MNG": "cst",
    "CLR": "clr",
    "REP": "cst",
    "other_ms": "non",
    "other_0": "non",
    "other_": "non",
    "other_n": "non",
    "other_manuscript": "non",
}

def get_elife_labels(row):
    """
    Takes a row from the labeled elife csv
    returns a dict in a standardize format
    """

    labels = {
        "pol": elife_pol_map[row["pol"]],
        "asp": elife_asp_map[row["asp"]],
        "epi": ("epi" if row["act"] in ["REQ", "EVL", "HYP"] else "nep"),
    }
    return labels


def preprocess_elife(data_dir):
    """
    Takes a csv of labeled elife reviews;
    Writes a json file formatted,
    1 line = 1 sentence with hand labels + meta data
    """
    print("Preprocessing labeled eLife data.")
    
    # read in csv of labels
    elife_csv = glob.glob(data_dir + "raw/eLife/*relabeled.csv")[0]
    elife_df = pd.read_csv(elife_csv)
    tr_dfs = []
    de_dfs = []
    te_dfs = []

    # stratified sampling by class of asp:
    for asp in elife_df["asp"].unique():
        df = elife_df[elife_df["asp"] == asp]
        DevTest = df.sample(frac=0.2, random_state=72)
        train_df = df.drop(DevTest.index)
        tr_dfs.append(train_df)
        dev_df = DevTest.sample(frac=0.5, random_state=72)
        de_dfs.append(dev_df)
        test_df = DevTest.drop(dev_df.index)
        te_dfs.append(test_df)

    dev_df = pd.concat(de_dfs)
    train_df = pd.concat(tr_dfs)
    test_df = pd.concat(te_dfs)
    dfs_dct = {"train": train_df, "dev": dev_df, "test": test_df}

    # output rows as json lines
    for task in "train dev test".split():
        lines = collections.defaultdict(list)
        for _, row in dfs_dct[task].iterrows():
            for feature, label in get_elife_labels(row).items():
                lines[feature].append(
                    {
                        "identifier": f"elife|{task}|{row['review_id']}",
                        "text": f"{row['text']}",
                        "label": label,
                    }
                )
        for feature, examples in lines.items():
            output_dir = f"{data_dir}/labeled/{feature}/{task}/"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/elife.jsonl", "w") as f:
                f.write("\n".join(json.dumps(e) for e in examples))


def get_unlabeled_elife_data(data_dir):
    """
    Summons all reviews from bigquery;
    Formats them by tokenizing them and appending meta data
    as a formatted dict, then
    Writes a giant json, 1 line = 1 review sentence to be predicted
    """
    print("Downloading unlabeled eLife data.")

    # GLOBALS
    BQ_CLIENT = bigquery.Client()
    QRY = """
    SELECT Reviewer_ID, review_id, Major_comments,
    FROM `gse-nero-dmcfarla-mimir`.eLife.eLife_Reviews_IDRating
    """

    # GET Data
    reviews_df = BQ_CLIENT.query(QRY).result().to_dataframe()
    reviews_df = reviews_df.dropna()

    # Write to json
    print("Processing unlabeled eLife data.")
    lines = []
    for _, row in tqdm(reviews_df.iterrows()):
        sentences = tokenize(row["Major_comments"])
        for i, sentence in enumerate(sentences):
            line = {
                "identifier": f"elife|predict|{row['review_id']}|{i}",
                "text": f"{sentence.text}",
                "label": None,
            }
            lines.append(line)
    for feature in "pol asp epi".split():
        output_dir = f"{data_dir}/unlabeled/{feature}/predict"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/elife.jsonl", "w") as f:
            f.write("\n".join(json.dumps(line) for line in lines))


            
# ==== GPT
    
def preprocess_gpt(data_dir):
    
    gpt_dct = {
        "iclr": {
            "train": train_iclr,
            "dev": dev_iclr, 
            "test": test_iclr
        },
        "elife": {
            "train": train_elife,
            "dev": dev_elife, 
            "test": test_elife
        },
    }
    
    
    for task in "train dev test".split():
        for feature in "pol asp".split():
            output_dir = f"{data_dir}/labeled/{feature}/{task}/"
            with open(f"{output_dir}/{}.jsonl", "w") as f:
                f.write("\n".join(json.dumps(example) for example in examples))

# ==== DISAPERE

disapere_pol_map = {"none": "non", "pol_negative": "neg", "pol_positive": "pos"}

# disapere_asp_map = {
#     "arg_other": "non",
#     "asp_clarity": "clr",
#     "asp_meaningful-comparison": "mng",
#     "asp_motivation-impact": "mot",
#     "asp_originality": "org",
#     "asp_replicability": "rep",
#     "asp_soundness-correctness": "snd",
#     "asp_substance": "sbs",
#     "none": "non",
# }
disapere_asp_map = {
    "arg_other": "non",
    "asp_clarity": "clr",
    "asp_meaningful-comparison": "cst",
    "asp_motivation-impact": "nvl",
    "asp_originality": "nvl",
    "asp_replicability": "cst",
    "asp_soundness-correctness": "acc",
    "asp_substance": "acc",
    "none": "non",
}

def get_disapere_labels(sent):
    labels = {
        "pol": disapere_pol_map[sent["polarity"]],
        "asp": disapere_asp_map[sent["aspect"]],
    }
    labels["epi"] = (
        "epi" if sent["review_action"] in ["arg_request", "arg_evaluative"] else "nep"
    )
    return labels


def preprocess_disapere(data_dir):
    print("Preprocessing labeled DISAPERE data.")
    for subset in "train dev test".split():
        lines = collections.defaultdict(list)
        for filename in glob.glob(f"{data_dir}/raw/disapere/{subset}/*.json"):
            with open(filename, "r") as f:
                obj = json.load(f)
                review_id = obj["metadata"]["review_id"]
                identifier_prefix = f"disapere|{subset}|{review_id}|"
                for sent in obj["review_sentences"]:
                    for task, label in get_disapere_labels(sent).items():
                        lines[task].append(
                            {
                                "identifier": f'{identifier_prefix}{sent["sentence_index"]}',
                                "text": sent["text"],
                                "label": label,
                            }
                        )
        for task, examples in lines.items():
            output_dir = f"{data_dir}/labeled/{task}/{subset}/"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/disapere.jsonl", "w") as f:
                f.write("\n".join(json.dumps(e) for e in examples))


# ==== AMPERE

ampere_epi_map = {
    "non-arg": "nep",
    "evaluation": "epi",
    "request": "epi",
    "fact": "nep",
    "reference": "nep",
    "quote": "nep",
}


def preprocess_ampere(data_dir):
    print("Preprocessing labeled AMPERE data.")
    examples = []

    for filename in glob.glob(f"{data_dir}/raw/ampere/*.txt"):
        print(filename)
        review_id = filename.split("/")[-1].rsplit(".", 1)[0].split("_")[0]
        with open(filename, "r") as f:
            sentence_dicts = []
            for i, line in enumerate(f):
                label, sentence = line.strip().split("\t", 1)
                examples.append(
                    {
                        "identifier": f"ampere|train|{review_id}|{i}",
                        "text": sentence,
                        "label": ampere_epi_map[label],
                    }
                )
    with open(f"{data_dir}/labeled/epi/train/ampere.jsonl", "w") as f:
        f.write("\n".join(json.dumps(e) for e in examples))


# ==== ReviewAdvisor

# revadv_label_map = {
#     "positive": "pos",
#     "negative": "neg",
#     "clarity": "clr",
#     "meaningful_comparison": "mng",
#     "motivation": "mot",
#     "originality": "org",
#     "replicability": "rep",
#     "soundness": "snd",
#     "substance": "sbs",
# }
revadv_label_map = {
    "positive": "pos",
    "negative": "neg",
    "clarity": "clr",
    "meaningful_comparison": "cst",
    "motivation": "nvl",
    "originality": "nvl",
    "replicability": "cst",
    "soundness": "acc",
    "substance": "acc",
}


def label_sentences(sentences, label_obj):
    labels = [list() for _ in range(len(sentences))]
    for label_start, label_end, label in label_obj:
        label_interval = Interval(label_start, label_end)
        for i, sentence in enumerate(sentences):
            if label_interval == sentence.interval:
                labels[i].append(label)
            elif (
                label_start > sentence.interval.upper_bound
                or label_end < sentence.interval.lower_bound
            ):
                pass
            else:
                overlap = sentence.interval & label_interval
                if overlap.upper_bound - overlap.lower_bound > TOLERANCE:
                    labels[i].append(label)
    return labels


def preprocess_revadv(data_dir):
    print("Preprocessing labeled ReviewAdvisor data.")
    with gzip.open(f"{data_dir}/raw/revadv/review_with_aspect.jsonl.gz", "r") as f:
        lines = collections.defaultdict(list)
        for line in f:
            obj = json.loads(line)
            identifier_prefix = f'revadv|train|{obj["id"]}|'
            sentences = tokenize(obj["text"])
            revadv_labels = label_sentences(sentences, obj["labels"])
            converted_labels = {}
            for i, (sentence, label_list) in enumerate(zip(sentences, revadv_labels)):
                if not label_list or "summary" in label_list:
                    converted_labels = {"epi": "nep", "pol": "non", "asp": "non"}
                else:
                    converted_labels["epi"] = "epi"
                    asp, pol = label_list[0].rsplit("_", 1)
                    converted_labels = {
                        "epi": "epi",
                        "pol": revadv_label_map[pol],
                        "asp": revadv_label_map[asp],
                    }
                    for task, label in converted_labels.items():
                        lines[task].append(
                            {
                                "identifier": f"{identifier_prefix}{i}",
                                "text": sentence.text,
                                "label": label,
                            }
                        )
    for task, examples in lines.items():
        with open(f"{data_dir}/labeled/{task}/train/revadv.jsonl", "w") as f:
            f.write("\n".join(json.dumps(e) for e in examples))


def prepare_unlabeled_iclr_data(data_dir):
    print("Preprocessing unlabeled ICLR data.")
    lines = collections.defaultdict(list)
    for filename in glob.glob(f"{data_dir}/raw/iclr/*.json"):
        with open(filename, "r") as f:
            obj = json.load(f)
            review_id = obj["identifier"]
            identifier_prefix = f"iclr|predict|{review_id}|"
            for i, sent in enumerate(tokenize(obj["text"])):
                for task in "epi pol asp".split():
                    lines[task].append(
                        {
                            "identifier": f"{identifier_prefix}{i}",
                            "text": sent.text,
                            "label": None,
                        }
                    )
    for task, examples in lines.items():
        output_dir = f"{data_dir}/unlabeled/{task}/predict/"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/iclr.jsonl", "w") as f:
            f.write("\n".join(json.dumps(e) for e in examples))


def main():

    args = parser.parse_args()
    print(args.data_dir)
    # preprocess_elife(args.data_dir)
    # get_unlabeled_elife_data(args.data_dir)
    preprocess_disapere(args.data_dir)
    # preprocess_ampere(args.data_dir)
    preprocess_revadv(args.data_dir)
    # print("Downloading ICLR data")
    # iclr_lib.get_iclr_data(f'{args.data_dir}/raw/iclr/')
    # prepare_unlabeled_iclr_data(args.data_dir)


if __name__ == "__main__":
    main()
