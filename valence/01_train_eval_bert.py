import argparse
import collections
import json
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import tqdm
import transformers

from contextlib import nullcontext
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

import classification_lib

TRAIN, EVAL, PREDICT = "train eval predict".split()

parser = argparse.ArgumentParser(
    description="Train BERT model for DISAPERE classification tasks"
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to directory containing train, dev and test jsons",
)
parser.add_argument(
    "-e",
    "--eval_dir",
    type=str,
    help="path to directory subset to evaluate",
)

parser.add_argument(
    "-m",
    "--mode",
    choices=(TRAIN, EVAL, PREDICT),
    type=str,
    help="train eval or predict",
)
parser.add_argument(
    "-t",
    "--task",
    type=str,
    help="train eval or predict",
)

# Hyperparameters
DEVICE = "cuda"
EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"

HistoryItem = collections.namedtuple(
    "HistoryItem", "epoch train_acc train_loss dev_acc dev_loss".split()
)

Example = collections.namedtuple("Example", "identifier text target".split())


# Wrapper around the tokenizer specifying the details of the BERT input
# encoding.
tokenizer_fn = lambda tok, text: tok.encode_plus(
    text,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding="max_length",
    max_length=512,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)


class ClassificationDataset(Dataset):
    """A torch.utils.data.Dataset for binary classification."""
    def __init__(self, data_dir, tokenizer, max_len=512):
        (
            self.identifiers,
            self.texts,
            self.target_indices,
        ) = classification_lib.get_text_and_labels(data_dir, get_labels=True)
        target_set = set(self.target_indices)
        assert list(sorted(target_set)) == list(range(len(target_set)))
        eye = np.eye(len(target_set), dtype=np.float64) # An identity matrix to easily switch to and from one-hot encoding.
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


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)
        if num_classes == 2:
            self.loss_fn = nn.BCEWithLogitsLoss() # Not sure if this is reasonable
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(bert_output["pooler_output"])
        return self.out(output)


def create_data_loader(data_dir, tokenizer):
    ds = ClassificationDataset(
        data_dir,
        tokenizer=tokenizer,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)


def build_data_loaders(data_dir, tokenizer):
    return (
        create_data_loader(
            f'{data_dir}/train/',
            tokenizer,
        ),
        create_data_loader(
            f'{data_dir}/dev/',
            tokenizer,
        ),
    )


def train_or_eval(
    mode,
    model,
    data_loader,
    device,
    return_preds=False,
    optimizer=None,
    scheduler=None,
):
    assert mode in [TRAIN, EVAL]
    is_train = mode == TRAIN
    if is_train:
        model = model.train()
        context = nullcontext()
        assert optimizer is not None
        assert scheduler is not None
    else:
        model = model.eval()
        context = torch.no_grad()

    results = []
    losses = []
    correct_predictions = 0
    n_examples = len(data_loader.dataset)

    with context:
        for d in tqdm.tqdm(data_loader):
            input_ids, attention_mask, targets, target_indices = [
                d[k].to(device)
                for k in "input_ids attention_mask targets target_indices".split()
            ]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            if return_preds:
                results.append((d["identifier"], preds.cpu().numpy().tolist()))
            loss = model.loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == target_indices)
            losses.append(loss.item())
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

    if return_preds:
        return results
    else:
        return correct_predictions.double().item() / n_examples, np.mean(losses)


def do_train(tokenizer, model, data_dir, ckpt_dir):
    """Train on train set, validating on dev set."""

    # We don't mess around with hyperparameters too much, just use decent ones.
    hyperparams = {
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "bert_model": PRE_TRAINED_MODEL_NAME,
    }

    (
        train_data_loader,
        dev_data_loader,
    ) = build_data_loaders(data_dir, tokenizer)

    # Optimizer and scheduler (boilerplatey)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    history = []
    best_accuracy = 0
    best_accuracy_epoch = None

    # EPOCHS is the maximum number of epochs we will run.
    for epoch in range(EPOCHS):

        # If no improvement is seen in PATIENCE iterations, we quit.
        if best_accuracy_epoch is not None and epoch - best_accuracy_epoch > PATIENCE:
            break

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        # Run train_or_eval ono train set in TRAIN mode, backpropagating
        train_acc, train_loss = train_or_eval(
            TRAIN,
            model,
            train_data_loader,
            DEVICE,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Run train_or_eval on dev set in EVAL mode
        dev_acc, dev_loss = train_or_eval(EVAL, model, dev_data_loader, DEVICE)

        # Recording metadata
        history.append(HistoryItem(epoch, train_acc, train_loss, dev_acc, dev_loss))
        for k, v in history[-1]._asdict().items():
            print(k + "\t", v)
        print()

        # Save the model parameters if this is the best model seen so far
        if dev_acc > best_accuracy:
            torch.save(model.state_dict(), f"{ckpt_dir}/best_bert_model.bin")
            best_accuracy = dev_acc
            best_accuracy_epoch = epoch

    with open(f"{ckpt_dir}/history.pkl", "wb") as f:
        pickle.dump(history, f)


def do_eval(tokenizer, model, data_dir, ckpt_dir):
    """Evaluate (on dev set?) without backpropagating."""
    test_data_loader = create_data_loader(
        data_dir,
        tokenizer,
    )

    # Get best model
    model.load_state_dict(torch.load(f"{ckpt_dir}/best_bert_model.bin"))

    dev_acc, dev_loss = train_or_eval(EVAL, model, test_data_loader, DEVICE)

    print("Dev accuracy", dev_acc)


def do_predict(tokenizer, model, data_dir, ckpt_dir):

    model.load_state_dict(torch.load(f"{ckpt_dir}/best_bert_model.bin"))

    predictions = {}
    with open(f"{data_dir}/features.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            encoded_review = tokenizer_fn(tokenizer, example["text"])
            input_ids = encoded_review["input_ids"].to(DEVICE)
            attention_mask = encoded_review["attention_mask"].to(DEVICE)

            output = model(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)
            predictions[example["identifier"]] = prediction.item()

    with open(f"{data_dir}/bert_predictions.jsonl", "w") as f:
        for identifier, pred in predictions.items():
            f.write(
                json.dumps(
                    {
                        "identifier": identifier,
                        "label": pred,
                    }
                )
                + "\n"
            )

def get_label_list(data_dir, task):
    with open(f'{data_dir}/metadata.json', 'r') as f:
        return json.load(f)['labels']

def make_checkpoint_path(data_dir, task):
    ckpt_dir = f"{data_dir}/{task}/ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir

def main():

    args = parser.parse_args()
    assert args.mode in [TRAIN, EVAL, PREDICT]

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    labels = get_label_list(args.data_dir, args.task)
    model = Classifier(len(labels)).to(DEVICE)
    model.loss_fn.to(DEVICE)

    ckpt_dir = make_checkpoint_path(args.data_dir, args.task)

    if args.mode == TRAIN:
        do_train(tokenizer, model, args.data_dir, ckpt_dir)
    elif args.mode == EVAL:
        do_eval(tokenizer, model, args.eval_dir, ckpt_dir)
    elif args.mode == PREDICT:
        do_predict(tokenizer, model, args.eval_dir, ckpt_dir)


if __name__ == "__main__":
    main()
