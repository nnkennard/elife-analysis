import argparse
import collections
import csv
import pickle
import torch
import torch.nn as nn
import transformers

from contextlib import nullcontext
from torch.optim import AdamW
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer

import who_wins_lib

seed = 34
torch.manual_seed(seed)
import random

random.seed(seed)
torch.cuda.manual_seed_all(seed)
import os

os.environ["PYTHONHASHSEED"] = str(seed)
import numpy as np

np.random.seed(seed)


DEVICE = "cuda"


parser = argparse.ArgumentParser(
    description="Evaluate BERT model for DISAPERE classification tasks"
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="path to overall data directory",
)

parser.add_argument(
    "-e",
    "--eval_subset",
    type=str,
    choices="train dev test predict".split(),
    help="subset to evaluate",
)


def argmax(l):
    return l.index(max(l))


def do_eval(tokenizer, model, config, eval_subset):
    """Evaluate without backpropagating."""
    data_loader = who_wins_lib.create_data_loader(
        config,
        eval_subset,
        tokenizer,
    )

    # Get best model
    model.load_state_dict(
        torch.load(f"checkpoints/{config.config_name}/best_bert_model.bin")
    )
    results = who_wins_lib.train_or_eval(
        who_wins_lib.EVAL, model, data_loader, DEVICE, return_probs=True
    )

    with open(f"results/{config.config_name}_{eval_subset}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow("identifier label true_label".split() + config.labels)
        
        for i in results:
            for identifier, probs, target in zip(*i):
                writer.writerow([identifier, argmax(probs), target] + probs)

#         for i in results:
#             for identifier, probs, target in zip(*i):
#                 # print(len(probs[0]))
#                 # print(len(probs[1]))
#                 if len(probs[0]) == 2:
#                     # RoBERTa output shape: (batch_size, 2)
#                     max_prob_indices = [np.argmax(p) for p in probs]
#                 else:
#                     # Electra output shape: (batch_size, sequence_length, 2)
#                     max_prob_indices = [np.argmax(p[:, 0]) for p in probs]

#                 predicted_labels = [
#                     int(idx) for idx in max_prob_indices
#                 ]  # Convert indices to predicted labels (0 or 1)

#                 max_prob = [p[idx] for p, idx in zip(probs, max_prob_indices)]  # Get the maximum probability for each example
#                 writer.writerow(
#                     [identifier, predicted_labels[0], target] + max_prob
#                 )

#             for i in results:
#                 identifiers, probs, targets = i
#                 for identifier, prob_list, target in zip(identifiers, probs, targets):
#                     max_prob_class1 = max(prob_list[0])  # Max probability of class 1
#                     max_prob_class2 = max(prob_list[1])  # Max probability of class 2
#                     predicted_label = np.argmax(prob_list)  # Predicted label (0 or 1)

#                     writer.writerow([identifier, predicted_label, target, max_prob_class1, max_prob_class2])


def main():
    args = parser.parse_args()

    config = who_wins_lib.read_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    
    # -----------------
    if config.model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
    # -----------------

    model = who_wins_lib.Classifier(len(config.labels), config.model_name).to(DEVICE)
    model.loss_fn.to(DEVICE)

    do_eval(tokenizer, model, config, args.eval_subset)


if __name__ == "__main__":
    main()
