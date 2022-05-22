# Similarity

We should be able to use similarity code for both salience (between review and manuscript) and consonance (between pairs of reviews).

Since we don't have gold labels for salience, I will test salience models on DISAPERE review-rebuttal pairs.

Input: json files in the format:

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

To produce this input format with DISAPERE, run

```
mkdir pdfs
python 00_prep_disapere_data.py -d /path/to/disapere/final_dataset/ -p pdfs/ -o disapere_intermediate.json
```

If review and/or rebuttal text is already sentence tokenized, separate sentences with `"\n\n"` to skip application of the sentence separation model.

Next, extract the text from each pdf. You may have to change paths to various grobid components below.
```
mkdir xmls/
java -Xmx4G -jar /path/to/grobid/grobid-0.7.1/grobid-core/build/libs/grobid-core-0.7.1-onejar.jar \
    -gH /path/to/grobid/grobid-0.7.1/grobid-home \
    -dIn pdfs/ \
    -dOut xmls/ \
    -exe processFullText
```

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
          "review_sentences": <review_sentences>
          "rebuttal_sentences": <optional_rebuttal_sentences>
        },
        
      ],
      
    }
  ],
  "manuscript_files": ... (same as before),
  "manuscript_sentences": [
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
python 01_tokenize.py -i disapere_intermediate.json -x xmls/ -d disapere_results
```

You can then run the similarity scripts. These only implement S-BERT similarity and Jaccard similarity for now.

```
mkdir disapere_results
python 02_rebuttal_review_salience.py -d disapere_results/
python 03_review_manuscript_salience.py -d disapere_results/
python 04_review_review_consonance.py -d disapere_results/
```
