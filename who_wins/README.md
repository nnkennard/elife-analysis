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

### 5. Training configurations
Training configurations are set up using yaml.
To train a new model, first set up a config. This allows you to swap in different models, and different combinations of datasets for train, dev and test (e.g. train on ICLR test on eLife, train on both test on eLife, etc)

