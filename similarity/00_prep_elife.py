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
	help="What's the path to the eLife review data",
)

arg_parser.add_argument(
	"--output_file", 
	"-o", 
	default="elife_similarity_input.json",
	type=str, 
	help="Name the output json file",
)

arg_parser.add_argument(
	"--data_source", 
	"-d", 
	default="csv", 
	type=str, 
	help="Data are from 'csv' or 'bq'? (BQ must be updated to have review text ids!)",
)

def gen_df_from_CSV(path): 
    """
    Reads csvs from Jupyter VM.
    """
    df = pd.read_csv(path+"elife_reviews_OLD.csv")
    return df

BQ_CLIENT = bigquery.Client()
REVIEW_QRY = """
  	SELECT Manuscript_no, Major_comments, Reviewer_ID
    FROM `gse-nero-dmcfarla-mimir`.eLife.eLife_Reviews_OldStyle
	"""

def gen_df_from_BQ():
    """
    Grabs data from BQ server.
    """
    df = (
        BQ_CLIENT.query(REVIEW_QRY)
        .result()
        .to_dataframe())
    return df 


def build_structures_CSV(df): 
    structures = []
    ms_dfs = df.groupby("Manuscript no.")
    for ms_no, ms_df in ms_dfs:
        r_count = 0
        reviews = []
        for review_i, review_df in ms_df.iterrows():
            r_count +=1
            # no review text id rn, making it up ad hoc
            if "review_id" in review_df: 
                review_id = review_df["review_df"]
            else:
                review_id = "{}_{}".format(ms_no, r_count)
            review_dict = {
            "review_id": review_id,
            "review_text": review_df["Major comments"],
            "reviewer_id": review_df["Reviewer ID"], 
            "rebuttal_text": None 
            }
            reviews.append(review_dict)
        structures.append([{"forum_id": ms_no, "reviews": reviews}])
    return structures


def build_structures_BQ(df):
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
            "rebuttal_text": None 
            }
            reviews.append(review_dict)
        structures.append([{"forum_id": ms_no, "reviews": reviews}])
    return structures


def main(): 
    args = arg_parser.parse_args()

    if args.data_source == "bq":
        print("Getting data from BQ server...")
        reviews_df = gen_df_from_BQ()
        structures = build_structures_BQ(reviews_df)
    else:
        reviews_df = gen_df_from_CSV(args.eLife_review_path)
        structures = build_structures_CSV(reviews_df)   
        
    with open(args.output_file, "w") as f:
        json.dump(
            {
                "structures": structures[:10],
            },
            f,
        )

if __name__ == "__main__":
    main()