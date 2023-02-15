import collections
from interval import Interval
import json
import sys
import stanza
import tqdm

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
                print(overlap.upper_bound - overlap.lower_bound)
                if overlap.upper_bound - overlap.lower_bound > TOLERANCE:
                    labels[i].append(label)
    return labels


def build_dict(review_id, sentence_idx, sentence, label_list):
    if "summary" in label_list:
        labels = {
            "review_action": "arg_structuring",
            "fine_review_action": "arg-structuring_summary",
            "aspect": "none",
            "polarity": "none",
        }
    elif not label_list:
        labels = {
            "review_action": "none",
            "fine_review_action": "none",
            "aspect": "none",
            "polarity": "none",
        }
    else:
        first_label = label_list.pop(0)
        assert "_" in first_label
        aspect, polarity = first_label.rsplit("_", 1)
        labels = {
            "review_action": "arg_evaluative",
            "fine_review_action": "none",
            "aspect": aspect,
            "polarity": polarity,
        }
    labels["sentence_index"] = sentence_idx
    labels["review_id"] = review_id
    labels["text"] = sentence.text
    return labels


def main():
    with open(sys.argv[1], "r") as f:
        lines = f.readlines()[:1000]
        for line in tqdm.tqdm(lines):
            obj = json.loads(line)
            sentences = tokenize(obj["text"])
            labels = label_sentences(sentences, obj["labels"])
            sentence_dicts = []
            for i, (sentence, label_list) in enumerate(zip(sentences, labels)):
                sentence_dicts.append(build_dict(obj["id"], i, sentence, label_list))

            with open(f'review_advisor_converted/{obj["id"]}.json', "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "review_id": obj["id"],
                        },
                        "review_sentences": sentence_dicts,
                    },
                    f,
                )


if __name__ == "__main__":
    main()
