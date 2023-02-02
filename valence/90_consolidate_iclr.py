import collections
import glob
import json
import sys


def main():

    input_jsons_dir, polarity_output_file = sys.argv[1:]
    paper_metadata = {}
    review_metadata = {}

    label_collector = collections.defaultdict(lambda: collections.defaultdict(dict))
    review_counts = collections.Counter()

    for task_name in "review_action polarity ms_aspect".split():
        filename = polarity_output_file.replace("polarity", task_name)
        with open(filename, "r") as f:
            for line in f:
                obj = json.loads(line)
                review, sentence_idx = obj["identifier"].split("|||")
                label_collector[review][int(sentence_idx)][task_name] = obj[
                    "prediction"
                ]

    for filename in glob.glob(f"{input_jsons_dir}/*.json"):
        with open(filename, "r") as f:
            obj = json.load(f)
            metadata = obj["metadata"]
            review_id = filename.split("/")[-1].split(".")[0]
            review_metadata[review_id] = {
              'rating': obj['rating'],
              'reviewer': obj['reviewer']
            }
            review_counts[metadata["forum_id"]] += 1
            if metadata["forum_id"] in paper_metadata:
                assert metadata == paper_metadata[metadata["forum_id"]]
            else:
                paper_metadata[metadata["forum_id"]] = metadata

    for forum_id, num_reviews in review_counts.items():
        forum_dict = {"forum_metadata": paper_metadata[forum_id], 'reviews': []}
        for i in range(num_reviews):
            review_id = f"{forum_id}___{i}"
            review_labels = label_collector[review_id]
            review_dict = dict(review_metadata[review_id])
            review_dict['labels'] = []
            for j in range(len(review_labels)):
                review_dict['labels'].append(review_labels[j])
            forum_dict['reviews'].append(review_dict)
        with open(f'iclr_iclrOnly_all/{forum_id}.json', 'w') as f:
          json.dump(forum_dict, f)


if __name__ == "__main__":
    main()
