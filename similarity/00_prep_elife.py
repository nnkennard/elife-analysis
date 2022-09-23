import json
import argparse
import re
import os
import random
from google.cloud import bigquery
from google.cloud import storage

arg_parser = argparse.ArgumentParser(
    description="Structure eLife reviews & MS file paths into JSON")

arg_parser.add_argument(
    "-p",
    "--pdf_dir",
    default="/home/jupyter/00_daniel/02_ms/ms_pdfs/",
    type=str,
    help="path to directory where MS pdfs will be saved",
)

arg_parser.add_argument(
    "--output_file",
    "-o",
    default="/home/jupyter/00_daniel/02_ms/ms_pdfs/elife_similarity_input.json",
    type=str,
    help="Name the output json file",
)

arg_parser.add_argument(
    "--n_comparisons",
    "-n",
    default=100,
    type=int,
    help="size of random sample of comparisons",
)

# Set seed
random.seed(777)

# Initialize google clients
BQ_CLIENT = bigquery.Client()
STORAGE_CLIENT = storage.Client()

# Add review_id + rating_hat when BQ is updated
REVIEW_QRY = """
    SELECT Manuscript_no, Major_comments, Reviewer_ID
    FROM `gse-nero-dmcfarla-mimir`.eLife.eLife_Reviews_OldStyle
    """


def summon_review_df():
    """
    Returns a pandas DF containing eLife reviews
    by summoning reviews from Google BQ.
    """
    df = (
        BQ_CLIENT.query(REVIEW_QRY)
        .result()
        .to_dataframe())
    df_nonans = df.dropna()
    return df_nonans


def gen_review_dicts(df):
    """
    Returns a list of dicts containing
    each MS's review metadata and review
    text (structured into a dict) by
    looping row-wise (review-wise) through
    a pandas DF grouped by MS

    Expected output (generic):
    [{"forum_id": int, "reviews": review_dict},
     {"forum_id": int, "reviews": review_dict},
     ...
    ]
    """
    # Initialize list to contain review data for all MSs
    all_review_dicts = []
    # loop through df, MS-wise
    for ms_no, ms_df in df.groupby("Manuscript_no"):
        # initialize list of current MS's own reviews
        ms_review_dicts = []
        # loop through each review (row) of current MS's own df
        r_count = 0
        for review_i, review_df in ms_df.iterrows():
            ##############################################
            # gen unique review id on the fly
            # delete when BQ is updated!
            r_count += 1
            if "review_id" in review_df:
                review_id = review_df["r_count"]
            else:
                review_id = "{}_{}".format(ms_no, r_count)
            ##############################################
            # organize current review's data into dict
            review_dict = {
                "review_id": review_id,  # delete when BQ is updated
                "review_text": review_df["Major_comments"],
                # "review_id": review_df["review_id"], # uncomment when BQ is updated
                "reviewer_id": review_df["Reviewer_ID"],
                "rebuttal_text": None
            }
            ms_review_dicts.append(review_dict)
        all_review_dicts.append({"forum_id": ms_no,
                                  "reviews": ms_review_dicts})
    return all_review_dicts


def summon_ms_pdfs(google_dir="mimir-elife-pdfs",
                   google_subdir="initial_submissions",
                   delimiter=None,
                   local_dir=None,
                   n=None):
    """
    Downloads MS pdfs from storage to local VM and
    Returns list of dicts containing all their
    local file paths (str)

    Args:
      google_dir: str name of google storage "bucket", the dir
      google_subdir: str name of google "prefix", the folder in the dir
      delimiter: str name of a sub-subdir; if None, all files in subdir returned
      local_dir: str of path where pdfs should be stored
      n: int sample size


    Note: MS ID is located in name of the dir
    containing its PDF (not in its filename)

    Expected output (generic):
    [{"forum_id": ms_no, "path": "path"},
     {"forum_id": ms_no, "path": "path"},
     ...
    ]
    """

    # get list of all blobs
    google_blobs = STORAGE_CLIENT.list_blobs(google_dir,
                                             prefix=google_subdir,
                                             delimiter=delimiter)

    # create gateway to bucket
    google_bucket = STORAGE_CLIENT.bucket(google_dir)

    # get filenames from blobs
    google_filenames = [blob.name for blob in google_blobs]
    google_filenames = random.choices(google_filenames, k=n)

    # initialize list of ms path dicts
    ms_path_dicts = []

    # loop through each blob (filename) and
    # initialize its blobbiness
    # extract ms no and
    # create new name and
    # download
    for filename in google_filenames:
        google_blob = google_bucket.blob(filename)
        ms_no = re.findall(r'eLife-(\d+)/', filename)[0]
        filename = local_dir + str(ms_no) + ".pdf"
        google_blob.download_to_filename(filename)
        ms_path_dicts.append({"forum_id": ms_no,
                              "manuscript_pdf_path": filename})
    return ms_path_dicts


def main():
    args = arg_parser.parse_args()

    print("Getting review df from BQ server...")
    reviews_df = summon_review_df()
    review_dicts = gen_review_dicts(reviews_df)

    print("Getting MS pdfs from BQ server...")
    ms_path_dicts = summon_ms_pdfs(local_dir=args.pdf_dir,
                                   n=args.n_comparisons)

    print("Writing JSON...")
    with open(args.output_file, "w") as f:
        json.dump(
            {
                "structures": review_dicts,
                "manuscript_files": ms_path_dicts,
            },
            f,
        )

    print("Completed.")


if __name__ == "__main__":
    main()
