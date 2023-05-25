import pandas as pd
import json 
import glob
import tqdm
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("-d", "--data_dir", default="", type=str, help="")
parser.add_argument("-c", "--corpus", default="", type=str, help="")

# Merge sentence-level predictions 
def merge_predictions(data_dir, corpus):
    print("Merging all predictions.")
    sentences_dct = {}
    for feature in "epi pol asp".split():
        infile = glob.glob(data_dir+f"{feature}/predict/{corpus}*predictions*")[0]
        with open(infile, 'r') as f: 
            for sentence in tqdm.tqdm(f):
                sentence_dct = json.loads(sentence)
                sent_id = sentence_dct['identifier']
                prediction = sentence_dct.get("prediction", None)
                if sent_id not in sentences_dct:
                    sentences_dct[sent_id] = {"labels": {"asp": None, 
                                                         "pol": None, 
                                                         "epi": None}}
                sentences_dct[sent_id]["labels"][f"{feature}"] = prediction
    return sentences_dct
            
# Aggregate up to review level
def agg_sentences(sentences_dct):
    print("Aggregating reviews.")
    reviews_dct = {}
    for key, val in tqdm.tqdm(sentences_dct.items()):
        review_id = key.split("|")[2]
        labels = val['labels']
        if review_id not in reviews_dct: 
            reviews_dct[review_id] = {}   
        if "len" not in reviews_dct[review_id]:
            reviews_dct[review_id]['len'] = 0
        reviews_dct[review_id]['len'] += 1
        if labels['epi'] == "epi":
            asp = f"{labels['pol']}_{labels['asp']}"
            if asp not in reviews_dct[review_id]:
                reviews_dct[review_id][asp] = 0
            reviews_dct[review_id][asp] += 1
    return reviews_dct
        
        
# Convert into pandas
def make_df(reviews_dct):
    print("Writing dataframe.")
    reviews_lst = []
    for review_id, variables in reviews_dct.items():
        dct = {}
        dct['review_id'] = review_id
        dct['ms_id'] = review_id.split("_")[0]
        dct.update(variables)
        reviews_lst.append(dct)
    reviews_df = pd.DataFrame.from_dict(reviews_lst)
    reviews_df = reviews_df.fillna(0)
    return reviews_df


def main(): 
    args = parser.parse_args()
    sentences = merge_predictions(args.data_dir, args.corpus)
    reviews = agg_sentences(sentences)
    df = make_df(reviews)
    print(df.shape)
    df.to_csv(args.data_dir+f"{args.corpus}_df.csv")
    
    
if __name__ == "__main__":
    main()