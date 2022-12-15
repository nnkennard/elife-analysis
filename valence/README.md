
This directory contains scripts to train and evaluate a model on [DISAPERE](https://github.com/nnkennard/DISAPERE), and use this model to predict labels for unlabeled peer review data.

## Data preparation

The `00_prep_disapere.py` script operates by iterating over all sentences in the DISAPERE dataset, converting each into a labeled example. The labels are different for review_action, aspect, etc, so we specify a **task** when running this script. Each **task** must have a label extractor defined in `00_prep_disapere.py` (see `polarity_exists()` and `review_action()` for examples). The created files are saved in `output_dir`, in the structure described below.

Example command:

```
python 00_prep_disapere.py -d ../../DISAPERE/DISAPERE -o disapere_data -t review_action
```

This script creates a subdirectory for the task in the data directory and files with sentences and their label indices for train/dev/test subsets. The metadata file contains a mapping from indices to label names, useful for making the prediction files human-readable later.

```
disapere_data/
└───review_action/
    │   metadata.json  
    │   train.jsonl
    │   dev.jsonl
    └───test.jsonl
```

## Training

The `01_train.py` can be used to train a BERT model on data formatted by `00_prep_disapere.py`. 

Example command:
```
python 01_train.py -d disapere_data/ -t review_action
```
This will train a model on the train set and evaluate it on the dev set. The best checkpoint is saved within the same directory as the data, along with a history of the run. In our example:

```
disapere_data/
└───review_action/
    └───ckpt/
    │   │   history.pkl 
    │   └───best_bert_model.bin
    │   metadata.json  
    │   train.jsonl
    │   dev.jsonl
    └───test.jsonl
```


## Evaluation

This script can be used to evaluate a model trained by `01_train.py` on the train, dev, or test set. Pass the name of the subset to be evaluated with the `-e` flag.

Example command:
```
python 02_eval.py -d disapere_data/ -t review_action -e train
```

## Preparing unlabeled data for prediction

This script converts json-formatted reviews into a sentence-separated format for classification. Example json files can be found in the `elife_mini/` directory.

Example command:
```
python 03_preprocess.py -r elife_mini/ -p preprocessed_elife_mini.jsonl
```

## Prediction

This script predicts sentence-level labels for a given task, with a file created by `03_preprocess.py` as input.

Example command:
```
python 04_predict.py -d disapere_data/ -t review_action -i preprocessed_elife_mini.jsonl
```

This produces the file `preprocessed_elife_mini_predictions.jsonl`.


## Label eLife

This script generates a hand-labeled eLife dataset with `-n_reviews` labeled through an interactive command line prompt. It summons the reviews from BQ, tokenizes their sentences, prints each sentence, and then asks the user to label the various aspects or arguments present in the sentence. It writes the labels of each sentence, sentence by sentence, at the specified local path. 

First, activate the conda env:
```
conda activate elife-sim
```

Example command of the very first time rating:
```
python 05_label_elife.py --file_path ~/my_labels.csv --n_reviews 1 --n_sents 2 --first_time true
```

Example command of returning to label:
```
python 05_label_elife.py -fp ~/my_labels.csv -nr 1 -ns 2 -ft false
```
This will return new, unlabeled sentences of the `nr` randomly sampled reviews and append these newly labeled ones to the ones you already labeled.

Notes of caution:
- If you put true for `ft` and your `file_path` is not new but contains previously labeled sentences, __this work will be overwritten.__
- If you make a typo, the script will either break or write the typo to the file. 
    - If the code breaks, the data from the sentence currently being labeled will be lost (others that were labeled before it are already saved). 
    - If the code does not break, the typo will be written to the file and you'll need to fix it by hand. Make a note of the identifier for easy re-labeling down the line.
- A smaller int for the arg `ns` means more breaks. It also means the reviews will be summoned from BQ more often (more queries).
- If you want to abort a current coding session mid-sentence, just put any non-int value into the prompt (the current sentence labels won't be written).
- Once we're confident that the script works, we can first put `n_reviews` to 10 for inter-rater reliability. We each will get the same randomly drawn sample of 10 reviews and all their sentences to label. Then, when we're ready to do more, we can each specifiy 100 or so. __Important__: for the purposes of validating the script and 