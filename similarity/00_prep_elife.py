import numpy as np 
import pandas as pd
import json 
import argparse 
import tqdm
from google.cloud import bigquery

arg_parser = argparse.ArgumentParser(
    description = "Structure eLife reviews & MS files into JSON") 

arg_parser.add_argument(
    "--eLife_review_path", 
    "-e", 
    default="~/00_daniel/00_reviews/00_data/",
    type=str, 
    help="What's the path to the eLife review data?",
)

arg_parser.add_argument(
    "--output_file", 
    "-o", 
    default="elife_similarity_input.json",
    type=str, 
    help="Name the output json file",
)

# arg_parser.add_argument(
#     "--data_source", 
#     "-d", 
#     default="csv", 
#     type=str, 
#     help="Data are from 'csv' or 'bq'? (BQ must be updated to have review text ids!)",
# )

# def get_reviews_VM(path): 
#     """
#     Creates a DF containing eLife reviews  
#     using flat files (CSVs) located on Jupyter VM.
#     """
#     df = pd.read_csv(path+"elife_reviews_OLD.csv")
#     return df

# def build_structures_CSV(df): 
#     structures = []
#     ms_dfs = df.groupby("Manuscript no.")
#     for ms_no, ms_df in ms_dfs:
#         r_count = 0
#         reviews = []
#         for review_i, review_df in ms_df.iterrows():
#             r_count +=1
#             # no review text id rn, making it up ad hoc
#             if "review_id" in review_df: 
#                 review_id = review_df["review_df"]
#             else:
#                 review_id = "{}_{}".format(ms_no, r_count)
#             review_dict = {
#             "review_id": review_id,
#             "review_text": review_df["Major comments"],
#             "reviewer_id": review_df["Reviewer ID"], 
#             "rebuttal_text": None 
#             }
#             reviews.append(review_dict)
#         structures.append([{"forum_id": ms_no, "reviews": reviews}])
#     return structures



BQ_CLIENT = bigquery.Client()
REVIEW_QRY = """
    SELECT Manuscript_no, Major_comments, Reviewer_ID
    FROM `gse-nero-dmcfarla-mimir`.eLife.eLife_Reviews_OldStyle
    """

def get_reviews():
    """
    Creates a DF containing eLife reviews  
    using by summoning reviews from Google BQ.
    """
    df = (
        BQ_CLIENT.query(REVIEW_QRY)
        .result()
        .to_dataframe())
    return df 


def get_ms_paths():
    """
    Gets list of all initial submissions.
    Organize by path and ID.
    """
    pass 


def build_structures(df):
    """
    BQ field names slight vary from csv's.
    """
    structures = []
    ms_dfs = df.groupby("Manuscript_no")
    for ms_no, ms_df in ms_dfs:
        r_count = 0
        reviews = []
        for review_i, review_df in ms_df.iterrows():
            r_count +=1 
            if "review_id" in review_df: 
                review_id = review_df["review_df"]
            else:
                review_id = "{}_{}".format(ms_no, r_count)
            review_dict = {
            "review_id": review_id,
                "review_text": review_df["Major_comments"],
                "reviewer_id": review_df["Reviewer_ID"], 
                "review_text": review_df["Major comments"],
                "reviewer_id": review_df["Reviewer ID"], 
            "rebuttal_text": None 
            }
            reviews.append(review_dict)
        structures.append([{"forum_id": ms_no, "reviews": reviews}])
    return structures


def main(): 
    args = arg_parser.parse_args()
        print("Getting data from BQ server...")
        reviews_df = get_reviews()
        structures = build_structures(reviews_df)
        
    with open(args.output_file, "w") as f:
        json.dump(
            {
                "structures": structures,
            },
            f,
        )

if __name__ == "__main__":
    main()