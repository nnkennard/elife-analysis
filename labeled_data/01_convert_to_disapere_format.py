import collections
import csv
import glob
import sys


INDICATOR_CONVERTER = {
    "arg_evaluative": ("review_action", "arg_evaluative"),
    "arg_request": ("review_action", "arg_request"),
    "arg_fact": ("review_action", "arg_fact"),
    "arg_structuring": ("review_action", "arg_structuring"),
    "arg_social": ("review_action", "arg_social"),
    "arg_other": ("review_action", "arg_other"),
    "asp_motivation-impact": ("aspect", "asp_motivation-impact"),
    "asp_originality": ("aspect", "asp_originality"),
    "asp_soundness-correctness": ("aspect", "asp_soundness-correctness"),
    "asp_substance": ("aspect", "asp_substance"),
    "asp_replicability": ("aspect", "asp_replicability"),
    "asp_meaningful-comparison": ("aspect", "asp_meaningful-comparison"),
    "asp_clarity": ("aspect", "asp_clarity"),
    "asp_other": ("aspect", "asp_other"),
    "req_edit": ("fine_review_action", "arg-request_edit"),
    "req_typo": ("fine_review_action", "arg-request_typo"),
    "req_experiment": ("fine_review_action", "arg-request_experiment"),
    "struc_summary": ("fine_review_action", "arg-structuring_summary"),
    "struc_heading": ("fine_review_action", "arg-structuring_heading"),
    "struc_quote": ("fine_review_action", "arg-structuring_quote"),
    "neg_polarity": ("polarity", "pol_negative"),
    "pos_polarity": ("polarity", "pol_positive"),
}

VALUE_CONVERTER = {
        "CLR": "asp_clarity",
        "EDT": "arg-request_edit",
        "EVL": "arg_evaluative",
        "EXP": "arg-request_experiment",
        "FCT": "arg_fact",
        "HDG": "arg-structuring_heading",
        "HYP": "arg_other",
        "MOT": "asp_motivation-impact",
        "NEG": "pol_negative",
        "ORG": "asp_originality",
        "POS": "pol_positive",
        "REQ": "arg_request",
        "SND": "asp_soundness-correctness",
        "SOC": "arg_social",
        "STR": "arg_structuring",
        "SUB": "asp_substance",
        "SUM": "arg-structuring_summary",
        "TYP": "arg-request_typo",
        }

KEY_CONVERTER  ={
        "act": "review_action",
        "asp": "aspect",
        "pol": "polarity",
        "req": "fine_review_action",
        "str": "fine_review_action",

        }

def get_labels_old_format(review_sentence_row):
    labels = {}
    for k, v in review_sentence_row.items():
        if k not in ["manuscript_no", "review_id", "identifier", "text"]:
            if v in ["", 'interpret', '.', 'intention', "'"]:
                if k == "arg_other":
                    labels["review_action"] = "arg_other"
                if k == "asp_other":
                    labels["aspect"] = "asp_other"

            elif int(v):
                labels.update(dict([INDICATOR_CONVERTER[k]]))
    return labels

def get_labels_new_format(review_sentence_row):
    labels = {}
    for k, v in review_sentence_row.items():
        if k not in ["manuscript_no", "review_id", "identifier", "text"]:
            if v == '0':
                continue
            elif v in ["other_tok_error", "other_interpret", "other_intepret", "other_other"]:
                assert k == 'act'
                labels["review_action"] = "arg_other"
            elif v in ["other_0"]:
                assert k == 'req'
                labels["fine_review_action"] = "arg-request_explanation"
            elif v in ['other_manuscript']:
                assert k == 'asp'
                labels['aspect'] = "none"
            else:    
                labels[KEY_CONVERTER[k]] = VALUE_CONVERTER[v]
    return labels

def compile_file(reader, annotator, data_format):
    review_objs = []
    by_review = collections.defaultdict(list)
    for row in reader:
        by_review[row["review_id"]].append(row)

    for review, rows in by_review.items():
        top_row = rows[0]
        metadata = {
            "review_id": row["review_id"],
            "forum_id": row["manuscript_no"],
            "reviewer": "Reviewer" + row["review_id"].split("_")[1],
            "annotator": annotator,
        }
        review_sentences = []
        for i, review_sentence_row in enumerate(rows):
            assert i == int(review_sentence_row["identifier"].split("|||")[-1])
            empty_sentence = {
                "review_id": review_sentence_row["review_id"],
                "sentence_index": i,
                "text": review_sentence_row["text"],
                "suffix": "",
                "review_action": "none",
                "fine_review_action": "none",
                "aspect": "none",
                "polarity": "none",
            }
            if data_format == "format0":
                empty_sentence.update(get_labels_old_format(review_sentence_row))
            else:
                assert data_format == "format1"
                empty_sentence.update(get_labels_new_format(review_sentence_row))
        review_objs.append({
            "metadata": metadata,
            "review_sentences": review_sentences
            })
    return review_objs
            
def compile_new_format(reader, annotator):
    pass


def main():
    import_dir = sys.argv[1]

    review_objs = []
    for filename in glob.glob(f"{import_dir}/*dss*format*"):
        source_file, annotator, data_format = (
            filename.split("/")[-1].split(".")[0].split("_")
        )
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            review_objs += compile_file(reader, annotator, data_format)
        print(len(review_objs))

if __name__ == "__main__":
    main()
