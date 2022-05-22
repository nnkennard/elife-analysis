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

If review and/or rebuttal text is already sentence tokenized, separate sentences with `"\n\n"` to skip application of the sentence separation model.


Baseline: just run similarity, sum over pairwise similarities of sentences. This shouldn't work too well.
Similarity measures:
* S-BERT?
* Arora?
