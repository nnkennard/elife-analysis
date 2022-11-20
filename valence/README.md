# DISAPERE classification

## Structure
This directory contains scripts to train and evaluate a model on [DISAPERE](https://github.com/nnkennard/DISAPERE), and use this model to predict labels for unlabeled peer review data.

The scripts include:

* `00_prep_disapere.py`: convert DISAPERE data into training files formatted for this model.
* `01_train_eval_predict.py`: operate a BERT model for classification (train/eval/predict)
* `02_prep_general_reviews.py`: convert general peer review text into files formatted for prediction with this model.

## Data preparation

The `00_prep_disapere.py` script operates by iterating over all sentences in the DISAPERE dataset, converting each into a labeled example. The labels are different for review_action, aspect, etc, so we specify a *task* when running this script. Each *task* must have a label extractor defined in the script (see `polarity_exists()` and `review_action()` for examples). The created files are saved in `output_dir`, in the structure described below.

Example command:
```
python 00_prep_disapere.py -d ../../DISAPERE/DISAPERE -t review_action -o disapere_data/
```

This script creates a subdirectory for the task in the data directory and files with sentences and their label indices for train/dev/test subsets. The metadata file contains a mapping from indices to label names, useful for making the prediction files human-readable later.

```
disapere_data/
└───review_action/
    │   metadata.json  
    │
    └───train/
    │   │   sentences.jsonl
    │   └── labels.jsonl
    │   
    └───dev/
    │   │   sentences.jsonl
    │   └── labels.jsonl
    │   
    └───test/
        │   sentences.jsonl
        └── labels.jsonl
```

## Training

The `01_train_eval_predict.py` can be used to train a BERT model on data formatted by `00_prep_disapere.py`. 


Example command:
```
python 01_train_eval_predict.py -d disapere_data/ -t review_action -m train
```

This will train a model on the train set (`disapere_data/train/*`) and evaluate it on the dev set. The best checkpoint is saved within the same director as the data, along with a history of the run. In our example:


```
disapere_data/
└───review_action/
    │   metadata.json  
    │
    └───train/
    │   │   sentences.jsonl
    │   └── labels.jsonl
    │   
    └───dev/
    │   │   sentences.jsonl
    │   └── labels.jsonl
    │   
    └───test/
    │   │   sentences.jsonl
    │   └── labels.jsonl
    │
    └───ckpt/
        │   best_bert_model.bin
        └── history.pkl
```

## Evaluation

For evaluation, the `01_train_eval_predict.py` uses the checkpoint created in `train` mode to evaluate various evaluation sets. Supply the name of the eval set (train/dev/test) with the `-e` flag.

Example command:

```
python 01_train_eval_predict.py -d disapere_data/ -t review_action -m eval -e train
```

This just prints out the result on the evaluation set.

## Preparing for prediction

To prepare reviews (such as eLife reviews) for classification, convert them into a json file with the following format:

```
[
  {
    "review_id": "placeholder_review_1",
    "review_text": "The authors convincingly demonstrate ..."
  },
  {
    "review_id": "placeholder_review_2",
    "review_text": "The data appear too preliminary ..."
  },
  {
    "review_id": "placeholder_review_3",
    "review_text": "Critical to the central points of..."
  }
]
```

Example command:

```
python 02_preprocess.py -i input_file.json -o predict_prepared_file.jsonl
```

## Prediction
For prediction, the `01_train_eval_predict.py` uses the checkpoint created in `train` mode to make predictions on the output file created above. Supply the name of the prediction file with the `-p` flag, and make sure to provide the task (with the `-t` flag).

Example command:

```
python 01_train_eval_predict.py -m predict -d disapere_data/ -t review_action -p predict_prepared_file.jsonl
```

This produces a new file in the same directory as the prepared file with the predicted labels.

