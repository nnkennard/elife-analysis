import argparse
import collections
import json
import numpy as np
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
    description="Train BERT model for DISAPERE classification tasks")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to directory containing train, dev and test jsons",
)
parser.add_argument(
    "-m",
    "--mode",
    choices=(TRAIN, EVAL, PREDICT),
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
    "HistoryItem", "epoch train_acc train_loss val_acc val_loss".split())

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

# An identity matrix to easily switch to and from one-hot encoding.
EYE_2 = np.eye(2, dtype=np.float64)


class PolarityDetectionDataset(Dataset):
  """A torch.utils.data.Dataset for binary classification.

    More info here: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """

  def __init__(self, data_dir, tokenizer, max_len=512):
    (
        self.identifiers,
        _,
        self.texts,
        self.target_indices,
    ) = classification_lib.get_features_and_labels(data_dir, get_labels=True)
    # For binary classification (BCEWithLogitsLoss) we need targets to be
    # one-hot encoded.
    self.targets = [EYE_2[int(i)] for i in self.target_indices]
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):  # This function is required
    return len(self.texts)

  def __getitem__(self, item):  # This function is required
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


class PolarityClassifier(nn.Module):
  """
    Basic binary classifier with BERT.
  """

  def __init__(self):
    super(PolarityClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, 2)

  def forward(self, input_ids, attention_mask): # This function is required
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.drop(bert_output["pooler_output"])
    return self.out(output)


def create_data_loader(data_dir, tokenizer):
  """Wrap a DataLoader around a PolarityDetectionDataset.

  While the dataset manages the content of the data, the data loader is more
  concerned with how the data is doled out, and is the connection between the
  dataset and the model.
  """

  ds = PolarityDetectionDataset(
      data_dir,
      tokenizer=tokenizer,
  )
  return DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4)


def build_data_loaders(data_dir, tokenizer):
  """Build train and dev data loaders from a structured data directory.

    TODO(nnk): Investigate why there is no test data loader.
  """
  assert "train" in data_dir
  return (
      create_data_loader(
          data_dir,
          tokenizer,
      ),
      create_data_loader(
          data_dir.replace("train", "dev"),
          tokenizer,
      ),
  )


def do_train(tokenizer, model, loss_fn, data_dir):
  """Train on train set, validating on validation set."""

  hyperparams = {
      "epochs": EPOCHS,
      "patience": PATIENCE,
      "learning_rate": LEARNING_RATE,
      "batch_size": BATCH_SIZE,
      "bert_model": PRE_TRAINED_MODEL_NAME,
  }

  (
      train_data_loader,
      val_data_loader,
  ) = build_data_loaders(data_dir, tokenizer)

  # Optimizer and scheduler (boilerplatey)
  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
  total_steps = len(train_data_loader) * EPOCHS
  scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=0, num_training_steps=total_steps)

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
        loss_fn,
        DEVICE,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Run train_or_eval on validation set in EVAL mode
    val_acc, val_loss = train_or_eval(EVAL, model, val_data_loader, loss_fn,
                                      DEVICE)

    # Recording metadata
    history.append(HistoryItem(epoch, train_acc, train_loss, val_acc, val_loss))
    for k, v in history[-1]._asdict().items():
      print(k + "\t", v)
    print()

    # Save the model parameters if this is the best model seen so far
    if val_acc > best_accuracy:
      torch.save(model.state_dict(), f"ckpt/best_bert_model.bin")
      best_accuracy = val_acc
      best_accuracy_epoch = epoch

  with open(f"ckpt/history.pkl", "wb") as f:
    pickle.dump(history, f)


def do_eval(tokenizer, model, loss_fn, data_dir):
  """Evaluate (on dev set?) without backpropagating."""
  assert "dev" in data_dir
  test_data_loader = create_data_loader(
      data_dir,
      tokenizer,
  )

  # Get best model
  model.load_state_dict(torch.load(f"ckpt/best_bert_model.bin"))

  dev_acc, dev_loss = train_or_eval(EVAL, model, test_data_loader, loss_fn,
                                    DEVICE)

  print("Dev accuracy", dev_acc)


def do_predict(tokenizer, model, data_dir):

  model.load_state_dict(torch.load(f"ckpt/best_bert_model.bin"))

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
      f.write(json.dumps({
          "identifier": identifier,
          "label": pred,
      }) + "\n")


def main():

  args = parser.parse_args()
  assert args.mode in [TRAIN, EVAL, PREDICT]

  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  model = PolarityClassifier().to(DEVICE)
  loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)

  if args.mode == TRAIN:
    do_train(tokenizer, model, loss_fn, args.data_dir)
  elif args.mode == EVAL:
    do_eval(tokenizer, model, loss_fn, args.data_dir)
  elif args.mode == PREDICT:
    do_predict(tokenizer, model, args.data_dir)


if __name__ == "__main__":
  main()
