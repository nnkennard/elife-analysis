# Similarity

We should be able to use similarity code for both salience (between review and manuscript) and consonaance (between reviews).

Since we don't have gold labels for salience, I will test salience models on DISAPERE review-rebuttal pairs.


Input: json files in the format:

```
{
  "review_sentences": [
    "review sentence 1",
    "review sentence 2", ...
  ],
  "rebuttal_sentences": [
    "rebuttal sentence 1",
    "rebuttal sentence 2", ...
  ],
  "review_rebuttal_alignment": [
    [
      0,
      0,
      1, ...
    ],
    [
      1,
      0,
      1, ...
    ]
  ]"manuscript_sentences": {
    "section 1 title": [
      "section 1 sentence 1",
      "section 1 sentence 2", ...
    ],
    "section 2 title": [
      "section 2 sentence 1",
      "section 2 sentence 2", ...
    ], ...
    
  }
}
```
