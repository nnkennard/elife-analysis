from contextlib import nullcontext
import collections
import io
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
import yaml
from transformers import BertModel, AutoModelForSequenceClassification
torch.cuda.empty_cache()
CONFIG_PATH = "configs/"

TRAIN, DEV, TEST, PREDICT, EVAL = "train dev test predict eval".split()

BATCH_SIZES = {
    TRAIN: 32,
    DEV: 32,
    TEST: 64,
    PREDICT: 64,
}

# Wrapper around the tokenizer specifying the details of the BERT input
# encoding.
tokenizer_fn = lambda tok, text: tok.encode_plus(
    text,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding="max_length",
    max_length=128,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)


Config = collections.namedtuple(
    "Config", "config_name task model_name train dev test predict labels".split()
)


def read_config(config_name, schema_path="schema.yml"):
    with open(schema_path, "r") as f:
        schema = yaml.safe_load(io.StringIO(f.read()))

    with open(f"{CONFIG_PATH}/{config_name}.yml", "r") as f:
        config = yaml.safe_load(io.StringIO(f.read()))
        assert config["config_name"] == config_name
        assert config["task"] in schema["tasks"]
        for dataset in config["train"]:
            assert config["task"] in schema["datasets"]["labeled"][dataset]
        return Config(labels=schema["labels"][config["task"]], **config)


def get_text_and_labels(config, subset):

    texts = []
    identifiers = []
    target_indices = []

    labeled_unlabeled = "unlabeled" if subset == "predict" else "labeled"

    for source in config._asdict()[subset]:
        with open(
            f"data/{labeled_unlabeled}/{config.task}/{subset}/{source}.jsonl", "r"
        ) as f:
            for line in f:
                example = json.loads(line)
                if example["label"] != "non":
                    texts.append(example["text"])
                    identifiers.append(example["identifier"])
                    if subset == "predict":
                        target_indices.append(-1)
                    else:
                        target_indices.append(config.labels.index(example["label"]))
                else:
                    pass
    return identifiers, texts, target_indices


class ClassificationDataset(Dataset):
    """A torch.utils.data.Dataset for classification."""

    def __init__(self, config, subset, tokenizer, max_len=512):
        (
            self.identifiers,
            self.texts,
            self.target_indices,
        ) = get_text_and_labels(config, subset)
        target_set = set(self.target_indices)
        assert target_set.issubset(range(len(config.labels))) or target_set == set([-1])
        eye = np.eye(
            len(config.labels), dtype=np.float64
        )  # An identity matrix to easily switch to and from one-hot encoding.
        self.targets = [eye[int(i)] for i in self.target_indices]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]

        encoding = tokenizer_fn(self.tokenizer, text)

        return {
            "reviews_text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.float64),
            "target_indices": self.target_indices[item],
            "identifier": self.identifiers[item],
        }


def create_data_loader(config, subset, tokenizer):
    """Wrap a DataLoader around a PolarityDetectionDataset.

    While the dataset manages the content of the data, the data loader is more
    concerned with how the data is doled out, and is the connection between the
    dataset and the model.
    """
    ds = ClassificationDataset(
        config,
        subset,
        tokenizer=tokenizer,
    )
    return DataLoader(ds, batch_size=BATCH_SIZES[subset], num_workers=4)


def build_data_loaders(config, tokenizer):
    """Build train and dev data loaders from a structured data directory."""
    return (
        create_data_loader(
            config,
            TRAIN,
            tokenizer,
        ),
        create_data_loader(
            config,
            DEV,
            tokenizer,
        ),
    )


class Classifier(nn.Module):
    def __init__(self, num_classes, model_name):
        super(Classifier, self).__init__()
        self.bert_like = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        if num_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss()  # Not sure if this is reasonable
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):  # This function is required
        return self.bert_like(input_ids=input_ids, attention_mask=attention_mask)
    

def train_or_eval(
    mode,
    model,
    data_loader,
    device,
    return_probs=False,
    optimizer=None,
    scheduler=None,
):
    """Do a forward pass of the model, backpropagating only for TRAIN passes."""
    assert mode in [TRAIN, EVAL]
    is_train = mode == TRAIN
    if is_train:
        model = model.train()  # Put the model in train mode
        context = nullcontext()
        # ^ This is so that we can reuse code between this mode and eval mode, when
        # we do have to specify a context
        assert optimizer is not None  # Required for backprop
        assert scheduler is not None  # Required for backprop
    else:
        model = model.eval()  # Put the model in eval mode
        context = torch.no_grad()  # Don't backpropagate

    results = []
    losses = []
    correct_predictions = 0
    n_examples = len(data_loader.dataset)

    with context:
        for d in tqdm.tqdm(data_loader):  # Load batchwise
            input_ids, attention_mask, targets, target_indices = [
                d[k].to(device)  # Move all this stuff to gpu
                for k in "input_ids attention_mask targets target_indices".split()
            ]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            # print(outputs)
            # ^ this gives logits
            _, preds = torch.max(outputs, dim=1)
            # TODO(nnk): make this argmax!
            if return_probs:
                # If this is being run as part of prediction, we need to return the
                # predicted indices. If we are just evaluating, we just need loss and/or
                # accuracy
                results.append(
                    (
                        d["identifier"],
                        nn.functional.softmax(outputs).cpu().numpy().tolist(),
                        target_indices.cpu().numpy().tolist(),
                        # preds.cpu().numpy().tolist(),
                    )
                )

            # Counting correct predictions in order to calculate accuracy later

            # ----------
            # print(preds.shape)
            # print(target_indices.shape)
            # if len(preds.shape) != len(target_indices):
            #     target_indices = target_indices.unsqueeze(1)
            # print(target_indices.shape)
            # ----------

            correct_predictions += torch.sum(preds == target_indices)

            if is_train:
                # We need loss for both train and eval, but restricting to
                # train in order to make our lives easier for predict mode
                
                # ----------
                # print(outputs.shape)
                # print(targets.shape)                
                # if len(outputs.shape) == 3: 
                #     targets = targets.unsqueeze(1)  # Add a dimension to match the input shape
                #     targets = targets.repeat(1, outputs.shape[1], 1)  # Repeat along the sequence length dimension
                # ----------

                loss = model.loss_fn(outputs, targets)


                losses.append(loss.item())
                # Backpropagation steps
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    if return_probs:
        return results
    else:
        # Return accuracy and mean loss
        return correct_predictions.double().item() / n_examples, np.mean(losses)
