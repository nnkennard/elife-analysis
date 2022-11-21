import argparse
import collections
import json
import pickle
import torch
import torch.nn as nn
import transformers

from contextlib import nullcontext
from torch.optim import AdamW
from transformers import BertTokenizer 

import classification_lib

DEVICE = "cuda"


parser = argparse.ArgumentParser(
    description="Predict DISAPERE labels using a BERT model"
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to overall data directory",
)

parser.add_argument(
    "-t",
    "--task",
    type=str,
    help="train eval or predict",
)

parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    help="preprocessed eLife file",
)

def do_predict(tokenizer, model, task_dir, input_file, labels):

  model.load_state_dict(torch.load(f"{task_dir}/ckpt/best_bert_model.bin"))

  predictions = {}
  with open(input_file, "r") as f:
    with open(input_file.replace(".jsonl", "_predictions.jsonl"), 'w') as g:
        for line in f:
          example = json.loads(line)
          encoded_review = classification_lib.tokenizer_fn(tokenizer, example["text"])
          input_ids = encoded_review["input_ids"].to(DEVICE)
          attention_mask = encoded_review["attention_mask"].to(DEVICE)
          output = model(input_ids, attention_mask)
          _, prediction = torch.max(output, dim=1)
          example['prediction'] = labels[prediction.item()]
          g.write(json.dumps(example) + "\n")


def main():
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained(classification_lib.PRE_TRAINED_MODEL_NAME)

    labels = classification_lib.get_label_list(args.data_dir, args.task)
    model = classification_lib.Classifier(len(labels)).to(DEVICE)
    model.loss_fn.to(DEVICE)

    task_dir = classification_lib.make_checkpoint_path(args.data_dir, args.task)

    do_predict(tokenizer, model, task_dir, args.input_file, labels)



if __name__ == "__main__":
    main()
