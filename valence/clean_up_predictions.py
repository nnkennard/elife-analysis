import collections
import json

FILE_MAP = {
    "action": "/home/jupyter/00_daniel/00_reviews/00_data/elife_review_jsons/preprocessed_eLife_predictions.jsonl",
    "aspect": "/home/jupyter/00_daniel/00_reviews/00_data/elife_review_jsons/preprocessed_eLife_ms_aspect_predictions.jsonl",
    "polarity": "/home/jupyter/00_daniel/elife-analysis/valence/elife_preprocessed_polarity_predictions.jsonl",
}


def get_label_map(filename):
    label_map = {}
    text_map = {}
    with open(filename, "r") as f:
        for line in f:
            obj = json.loads(line)
            label_map[obj["identifier"]] = obj["prediction"]
            text_map[obj["identifier"]] = obj["text"]
    return label_map, text_map


def main():

    label_map_map = {}
    for label_type, filename in FILE_MAP.items():
        print(label_type)
        label_map_map[label_type], text_map = get_label_map(filename)

    assert label_map_map["polarity"].keys() == label_map_map["aspect"].keys()
    assert label_map_map["action"].keys() == label_map_map["aspect"].keys()

    combined_map = {}
    for i, identifier in enumerate(sorted(label_map_map["action"].keys())):
        values = [x[identifier] for x in label_map_map.values()]
        combined_map[identifier] = values + [text_map[identifier]]

    sentences_by_review = collections.defaultdict(list)
    for identifier, values in combined_map.items():
        review_id, sentence_id = identifier.split("|||")
        sentences_by_review[review_id].append((int(sentence_id), values))

    for k, v in sentences_by_review.items():
        print(k)
        for i in sorted(v):
            print(i)
        print()
        break


if __name__ == "__main__":
    main()
