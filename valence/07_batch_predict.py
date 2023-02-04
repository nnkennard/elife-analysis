import argparse
import collections
import json
import pickle
import random
import torch
import torch.nn as nn
import transformers
import tqdm
from contextlib import nullcontext
from torch.optim import AdamW
from transformers import BertTokenizer

import classification_lib

DEVICE = "cuda"

seed = 34
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True

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
    "-e",
    "--expt_name",
    type=str,
    help="name for the experiment",
)

parser.add_argument(
    "-t",
    "--task",
    type=str,
    help="review_action or ms_aspect",
)

parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    help="preprocessed eLife file",
)

BATCH_SIZE = 128

predict_tokenizer_fn = lambda tok, texts: tok.batch_encode_plus(
    texts,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding="max_length",
    max_length=128,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)


def do_predict(tokenizer, model, task_dir, input_file, task, labels, expt_name):

    model.load_state_dict(torch.load(f"{task_dir}/ckpt/best_bert_model.bin"))

    with open(input_file, "r") as f:
        examples = [json.loads(line) for line in f]
        predictions = []
        for i in tqdm.tqdm(range(0, len(examples), BATCH_SIZE)):
            batch = [e["text"] for e in examples[i : i + BATCH_SIZE]]
            encoded_review = predict_tokenizer_fn(tokenizer, batch)
            input_ids = encoded_review["input_ids"].to(DEVICE)
            attention_mask = encoded_review["attention_mask"].to(DEVICE)
            output = model(input_ids, attention_mask)
            _, batch_predictions = torch.max(output, dim=1)
            predictions += batch_predictions
    for example, prediction in zip(examples, predictions):
        example["prediction"] = labels[prediction.item()]
    with open(
        input_file.replace(".jsonl", f"_{task}_{expt_name}_predictions.jsonl"), "w"
    ) as g:
        g.write("\n".join(json.dumps(e) for e in examples))


def main():
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(classification_lib.PRE_TRAINED_MODEL_NAME)

    labels = classification_lib.get_label_list(args.data_dir, args.task)
    model = classification_lib.Classifier(len(labels)).to(DEVICE)
    model.loss_fn.to(DEVICE)

    task_dir = classification_lib.make_checkpoint_path(args.data_dir, args.task)

    do_predict(
        tokenizer, model, task_dir, args.input_file, args.task, labels, args.expt_name
    )


if __name__ == "__main__":
    main()
