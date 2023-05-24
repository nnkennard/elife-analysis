import argparse
import collections
import json
import pickle
import random
import torch
import torch.nn as nn
import transformers
import tqdm
#from contextlib import nullcontext
#from torch.optim import AdamW
# from transformers import BertTokenizer, RobertaTokenizer
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer

import who_wins_lib

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




parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    help="preprocessed eLife file",
)

# BATCH_SIZE = 128
BATCH_SIZE = 64

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


def do_predict(tokenizer, model, input_file, config):

    model.load_state_dict(
        torch.load(f"checkpoints/{config.config_name}/best_bert_model.bin")
    )

    with open(input_file, "r") as f:
        examples = [json.loads(line) for line in f]
        predictions = []
        for i in tqdm.tqdm(range(0, len(examples), BATCH_SIZE)):
            batch = [e["text"] for e in examples[i : i + BATCH_SIZE]]
            encoded_review = predict_tokenizer_fn(tokenizer, batch)
            input_ids = encoded_review["input_ids"].to(DEVICE)
            attention_mask = encoded_review["attention_mask"].to(DEVICE)
            output = model(input_ids, attention_mask).logits
            _, batch_predictions = torch.max(output, dim=1)
            predictions += batch_predictions
    for example, prediction in zip(examples, predictions):
        example["prediction"] = config.labels[prediction.item()]
    with open(
        input_file.replace(".jsonl", f"_{config.config_name}_predictions.jsonl"), "w"
    ) as g:
        g.write("\n".join(json.dumps(e) for e in examples))


def main():
    args = parser.parse_args()

    config = who_wins_lib.read_config(args.config)
    
    
    # -----------------
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # if config.model_name == "roberta-base":
    #   tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    # else:
    #   tokenizer = BertTokenizer.from_pretrained(config.model_name)
    # -----------------


    model = who_wins_lib.Classifier(len(config.labels), config.model_name).to(DEVICE)
    model.loss_fn.to(DEVICE)

    do_predict(
        tokenizer, model, args.input_file, config 
    )


if __name__ == "__main__":
    main()
