
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
