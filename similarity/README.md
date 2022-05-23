# Similarity

We should be able to use similarity code for both salience (between review and manuscript) and consonance (between pairs of reviews).

Since we don't have gold labels for salience, we'll test salience models on DISAPERE review-rebuttal pairs.

## Preparing the environment
First, make sure you have the requirements installed in your virtual env:
```
python -m pip install -r requirements.txt
python -c "import stanza; stanza.download('en')"
```

## Preparing input
These scripts require input as a json file, with the format:

```
{
  "structures": [
    {
      "forum_id": <forum_id>,
      "reviews": [
        {
          "review_id": <review_id>,
          "review_text": <review_text>,
          "reviewer_id": <reviewer_id>,
          "rebuttal_text": <optional_rebuttal_text>
        },
        
      ],
      
    }
  ],
  "manuscript_files": [
    {
      "forum_id": <forum_id>,
      "manuscript_pdf_path": <path_to_manuscript_pdf>,
      
    },
    
  ]
}
```

To produce this input format with DISAPERE, run:

```
mkdir pdfs
python 00_prep_disapere_data.py -d /path/to/disapere/final_dataset/ -p pdfs/ -o intermediate.json
```
If review and/or rebuttal text happens to already be sentence tokenized, separate sentences with `"\n\n"` to skip application of the sentence separation model.

## Extracting text from pdfs
Next, extract the text from each pdf. You may have to change paths to various grobid components below.
```
mkdir xmls/
java -Xmx4G -jar /path/to/grobid/grobid-0.7.1/grobid-core/build/libs/grobid-core-0.7.1-onejar.jar \
    -gH /path/to/grobid/grobid-0.7.1/grobid-home \
    -dIn pdfs/ \
    -dOut xmls/ \
    -exe processFullText
```

## Tokenize
Then, tokenize the various texts. (By 'tokenize', we mean just separate sentences.) The tokenization script results in a modified file, with the format:
```
{
  "structures": [
    {
      "forum_id": <forum_id>,
      "reviews": [
        {
          "review_id": <review_id>,
          "review_text": <review_text>,
          "reviewer_id": <reviewer_id>,
          "rebuttal_text": <optional_rebuttal_text>
          "review_sentences": <review_sentences> (new!)
          "rebuttal_sentences": <optional_rebuttal_sentences> (new!)
        },
        
      ],
      
    }
  ],
  "manuscript_files": ... (same as before),
  "manuscript_sentences": [ (new!)
    {
      "forum_id": <forum_id>,
      "manuscript_sentences": {
        <section_1_name>: 
      },
      
    },
}
```

To run the tokenization script:
```
mkdir disapere_results/
python 01_tokenize.py -i intermediate.json -x xmls/ -o disapere_results
```

## Calculate similarities
You can then run the similarity scripts. These only implement S-BERT similarity and Jaccard similarity for now.

```
mkdir disapere_results
python 02_rebuttal_review_salience.py -o disapere_results/
python 03_review_manuscript_salience.py -o disapere_results/
python 04_review_review_consonance.py -o disapere_results/
```

These will result in the following reports being produced in the output directory:
```
<output_dir_name>/
  ├── rebuttal_review_salience.json
  ├── review_manuscript_salience.json
  └── review_review_consonance.json
```
