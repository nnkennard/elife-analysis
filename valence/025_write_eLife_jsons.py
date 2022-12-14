import json
import argparse
from google.cloud import bigquery
from google.cloud import storage
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Structure eLife reviews into JSONs")

parser.add_argument(
    "--output_dir",
    "-o",
    type=str,
    help="Name the output directory",
)


# Initialize google clients
BQ_CLIENT = bigquery.Client()
STORAGE_CLIENT = storage.Client()
REVIEW_QRY = """
    SELECT review_id, Major_comments, rating_hat,
    FROM `gse-nero-dmcfarla-mimir`.eLife.eLife_Reviews_IDRating
    """


def summon_reviews():
    """
    Returns a pandas DF containing eLife reviews
    by summoning reviews from Google BQ.
    """
    df = BQ_CLIENT.query(REVIEW_QRY).result().to_dataframe()
    df_nonans = df.dropna()
    return df_nonans


def write_jsons(df, output_dir):
    """
    Writes a json for each review.
    """
    for i, row in tqdm(df.iterrows()):
        identifier = row["review_id"]
        text = row["Major_comments"]
        j = {"identifier": identifier, "text": text}
        j_string = json.dumps(j)
        with open(output_dir + f"{identifier}.json", "w") as f:
            f.write(j_string)


def main():
    args = parser.parse_args()

    print("Summoning eLife reviews from BQ...\n")
    df = summon_reviews()
    print("\u2713", "Reviews loaded!")
    print("Writing eLife jsons")
    write_jsons(df, args.output_dir)
    print("\u2713", "jsons written!")


if __name__ == "__main__":
    main()
