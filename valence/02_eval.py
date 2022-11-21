import argparse
import collections
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
    description="Train BERT model for DISAPERE classification tasks"
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="path to overall data directory",
)

parser.add_argument(
    "-e",
    "--eval_subset",
    type=str,
    choices="train dev test".split(),
    help="subset to evaluate",
)


parser.add_argument(
    "-t",
    "--task",
    type=str,
    help="train eval or predict",
)

def do_eval(tokenizer, model, task_dir, eval_subset):
  """Evaluate (on dev set?) without backpropagating."""
  data_loader = classification_lib.create_data_loader(
          task_dir,
          eval_subset,
      tokenizer,
  )

  # Get best model
  model.load_state_dict(torch.load(f"{task_dir}/ckpt/best_bert_model.bin"))
  acc, loss = classification_lib.train_or_eval(classification_lib.EVAL, model, data_loader, DEVICE)

  print(f'Accuracy: {acc} Loss: {loss}')



def main():
    args = parser.parse_args()
    
    tokenizer = BertTokenizer.from_pretrained(classification_lib.PRE_TRAINED_MODEL_NAME)

    labels = classification_lib.get_label_list(args.data_dir, args.task)
    model = classification_lib.Classifier(len(labels)).to(DEVICE)
    model.loss_fn.to(DEVICE)

    task_dir = classification_lib.make_checkpoint_path(args.data_dir, args.task)

    do_eval(tokenizer, model, task_dir, args.eval_subset)



if __name__ == "__main__":
    main()
