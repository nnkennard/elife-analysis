# New pipeline for Which Values ASR version

## Changes from prior versions of the pipeline:
1. New task -- epi/nep
2. Task structure listed in `schema.yml`
3. Standard format of datasets
5. Training configurations
6. 
3. Outputs include model confidences

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
  │     └─── <dataset_name>/ # official dataset name from schema.yml
  │           └─── # Whatever raw format the dataset is provided in
  └─── labeled/
        └─── <task_name>/ # official task name from schema.yml 
              └─── <subset>/ # train/dev/test
                    │    <dataset_name_1>.jsonl
                    │    <dataset_name_2>.jsonl
                    └─── ...
```

### 5. Training configurations
Training configurations are set up using yaml.
To train a new model, first set up a config. This allows you to swap in different models, and different combinations of datasets for train, dev and test (e.g. train on ICLR test on eLife, train on both test on eLife, etc)

