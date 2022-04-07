import argparse
import collections
import glob
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

parser = argparse.ArgumentParser(description="Create examples for WS training")
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to data file containing score jsons",
)

parser.add_argument(
    "-n",
    "--dataset_name",
    type=str,
    help="e.g. disapere or ape",
)

DEVICE = "cuda"
EPOCHS = 100
PATIENCE = 20
TRAIN, EVAL = "train eval".split()
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"

HistoryItem = collections.namedtuple(
    "HistoryItem", "epoch train_acc train_loss val_acc val_loss".split())

Example = collections.namedtuple("Example",
  "identifier text target".split())

def get_label(original_label):
  return (0 if original_label == "none" else 1)

class PolarityDetectionDataset(Dataset):

  def __init__(self, data_dir, tokenizer, max_len=512):
    examples = []
    print(data_dir)
    for filename in sorted(glob.glob(f"{data_dir}/*")):
      with open(filename, 'r') as f:
        obj = json.load(f)
        review_id = obj["metadata"]["review_id"]
        for i, review_sentence in enumerate(obj["review_sentences"]):
          examples.append(Example(f"{review_id}|{i}", review_sentence["text"],
          get_label(review_sentence["pol"])))
    self.identifiers, self.texts, self.target_indices = zip(*examples)
    self.targets = [np.eye(2, dtype=np.float64)[int(i)] for i in self.target_indices]
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    text = str(self.texts[item])
    target = self.targets[item]


    encoding = self.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    return {
        "reviews_text": text,
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
        "targets": torch.tensor(target, dtype=torch.float64),
        "target_indices": self.target_indices[item],
        "identifier": self.identifiers[item],
    }


class SentimentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    output = self.drop(bert_output["pooler_output"])
    return self.out(output)


def create_data_loader(data_dir, subset, tokenizer):
  ds = PolarityDetectionDataset(
      f'{data_dir}/{subset}/',
      tokenizer=tokenizer,
  )


def build_data_loaders(data_dir, tokenizer):
  return (
      create_data_loader(
          data_dir,
          "train",
          tokenizer,
      ),
      create_data_loader(
          data_dir,
          "dev",
          tokenizer,
      ),
    )

def train_or_eval(
    mode,
    model,
    data_loader,
    loss_fn,
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
      loss = loss_fn(outputs, targets)
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


def get_metric_helper(data_dir, subset_key):
  with open(f"{data_dir}/examples_{subset_key}_helper.json", "r") as f:
    return json.load(f)


def main():

  args = parser.parse_args()

  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  (
      train_data_loader,
      val_data_loader,
  ) = build_data_loaders(args.data_dir, tokenizer)

  model = SentimentClassifier(2).to(DEVICE)
  optimizer = AdamW(model.parameters(), lr=2e-5)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  loss_fn = nn.BCEWithLogitsLoss().to(DEVICE)

  history = []
  best_accuracy = 0
  best_accuracy_epoch = None

  for epoch in range(EPOCHS):

    if best_accuracy_epoch is not None and epoch - best_accuracy_epoch > PATIENCE:
      break

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_or_eval(
        TRAIN,
        model,
        train_data_loader,
        loss_fn,
        DEVICE,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    val_acc, val_loss = train_or_eval(EVAL, model, val_data_loader, loss_fn,
                                      DEVICE)

    history.append(HistoryItem(epoch, train_acc, train_loss, val_acc, val_loss))
    for k, v in history[-1]._asdict().items():
      print(k + "\t", v)
    print()

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), f"outputs/best_model.bin")
      best_accuracy = val_acc
      best_accuracy_epoch = epoch

  with open(f"outputs/history.pkl", "wb") as f:
    pickle.dump(history, f)


if __name__ == "__main__":
  main()
