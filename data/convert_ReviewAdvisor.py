import json
import sys
import stanza
import tqdm

SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")
TOLERANCE = 3


def main():
  with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
      obj = json.loads(line)
      doc = SENTENCIZE_PIPELINE(obj['text'])
      sentence_pool  = {}
      for sentence in doc.sentences:
        start = sentence.to_dict()[0]['start_char']
        end = sentence.to_dict()[-1]['end_char']
        sentence_pool[(start, end)] = sentence.text


      starts, ends = zip(*sentence_pool.keys())

      for label_start, label_end, label in obj['labels']:
        start_adjustment, end_adjustment = 0, 0
        while label_start - start_adjustment not in starts:
          start_adjustment += 1
        while label_end + end_adjustment not in ends:
          end_adjustment += 1


        label_start -= start_adjustment
        label_end += end_adjustment

        if start_adjustment > 100 or end_adjustment > 100:
          print(obj['id'], start_adjustment, end_adjustment)

        found = False
        for (start, end), sentence_text in sentence_pool.items():
          if start >= label_start and end <= label_end:
            found = True
        assert found


if __name__ == "__main__":
  main()

