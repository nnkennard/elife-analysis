# New pipeline for Which Values ASR version

## Changes from prior versions of the pipeline:
1. New task -- epi/nep
2. Task structure listed in `schema.yml`
3. Standard format of datasets
4. Training configurations
5. Outputs include model confidences

### 1. New task: epi/nep
This task distinguishes between 'epistemic' sentences (request and evaluative) and 'non-epistemic' sentences (everything else). This allows us to share information between the two types of epistemic classes, and not waste model parameters on distinguishing between things like structure/social/other.

### 2. Task structure listed in `schema.yml`

Task information, including

* Which tasks are available
* Which datasets contain labels for which tasks
* What is the label set for each task
* What are the nice (printable) names for each label

is all available in a human- and machine-readable format in `schema.yml`

### 3. Standard format of datasets

In `01_preprocess_data.py`, all datasets are converted to a single format, using tasks and labels from `schema.yml`.

```
data/
  └─── raw/
  │     └─── <dataset_name>/ # official labeled dataset name from schema.yml
  │           └─── # Whatever raw format the dataset is provided in
  └─── labeled/
  │     └─── <task_name>/ # official task name from schema.yml 
  │           └─── <subset>/ # train/dev/test
  │                 │    <dataset_name_1>.jsonl
  │                 │    <dataset_name_2>.jsonl
  │                 └─── ...
  └─── unlabeled/
        └─── <task_name>/ # official task name from schema.yml 
              └─── predict/
                    │    <dataset_name_1>.jsonl # official unlabeled dataset name
                    │    <dataset_name_2>.jsonl
                    └─── ...
```

Each line in a _labeled_ jsonl file consists of a json-formatted dictionary with the fields `identifier`, `text`, and `label`. An example:

```
courbet $ head -n5 data/labeled/epi/train/disapere.jsonl
{"identifier": "disapere|train|B1euHOqi37|0", "text": "The present paper proposes a fast approximation to the softmax computation when the number of classes is very large.", "label": "nep"}
{"identifier": "disapere|train|B1euHOqi37|1", "text": "This is typically a bottleneck in deep learning architectures.", "label": "nep"}
{"identifier": "disapere|train|B1euHOqi37|2", "text": "The approximation is a sparse two-layer mixture of experts.", "label": "nep"}
{"identifier": "disapere|train|B1euHOqi37|3", "text": "The paper lacks rigor and the writing is of low quality, both in its clarity and its grammar.", "label": "epi"}
{"identifier": "disapere|train|B1euHOqi37|4", "text": "See a list of typos below.", "label": "nep"}
```

Lines in a _predict_ jsonl file are similar, but with `label` set to `null`:
```
courbet $ head -n2 data/unlabeled/epi/predict/iclr.jsonl
{"identifier": "iclr|predict|--GJkm7nt0___0|0", "text": "A knowledge distillation framework is proposed for efficient object recognition.", "label": null}
{"identifier": "iclr|predict|--GJkm7nt0___0|1", "text": "In this framework, the teacher network (TN) performs high accuracy prediction while two student networks (SN) mimic the prediction from TN.", "label": null}
```

Ideally, we wouldn't have multiple copies of the same unlabeled file under different directories for different tasks, but I did not think it was worth the time it would take to refactor it.

### 4. Training configurations
Training configurations are set up using yaml.
To train a new model, first set up a config. This allows you to swap in different models, and different combinations of datasets for train, dev and test (e.g. train on ICLR test on eLife, train on both test on eLife, etc)

[Example](https://github.com/nnkennard/elife-analysis/blob/main/who_wins/configs/C2.yml)

### 5. Output includes model confidences
Model output for all the datasets listed under `dev` or `test` in the config take the form of csv files in the `results/` directory. These include the predicted label, as well as the probabilities of all the different labels (the predicted label is just the one with the highest probability)

```
identifier,label,true_label,epi,nep
disapere|dev|SJgMKr5h3X|7,0,0,0.9999125003814697,8.749817061470821e-05
disapere|dev|SJgMKr5h3X|8,0,0,0.9989770650863647,0.001022971817292273
disapere|dev|SJgMKr5h3X|9,0,0,0.99992835521698,7.167855801526457e-05
disapere|dev|SJgMKr5h3X|10,0,0,0.999904990196228,9.505140042165294e-05
```

This information can be used, for example, to only apply apsect or polarity labels to sentences that are 'epistemic' with a probability higher than some threshold.

## Training new models

1. Create a config file in the `configs/` directory. Make sure that the name field of the config matches the filename.
2. Ensure that the required datasets have been downloaded in `00_prep_data.sh` (if necessary) and preprocessed using `01_preprocess_data.py`.
3. `python 02_train.py -c <config_name>`
4. `python 03_eval.py -c <config_name> -e <subset_to_evaluate on>` 
