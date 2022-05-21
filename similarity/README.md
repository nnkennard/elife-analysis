# Similarity

We should be able to use similarity code for both salience (between review and manuscript) and consonance (between reviews).

Since we don't have gold labels for salience, I will test salience models on DISAPERE review-rebuttal pairs.


Input: json files in the format:

```
{
  "review_id" : <review_id>,
  "manuscript_id": <manuscript_id>,
  "review_sentences": [
    <review_sentence_1>,
    <review_sentence_2>, ...
  ],
  "rebuttal_sentences": [
    <rebuttal_sentence_1>,
    <rebuttal_sentence_2>, ...
  ],
  "review_rebuttal_alignment": <alignment_array>
  ]"manuscript_sentences": {
    <section_1_title>: [
      <section_1_sentence_1>,
      <section_1_sentence_2>, ...
    ],
    <section_2_title>: [
      <section_2_sentence_1>,
      <section_2_sentence_2>, ...
    ], ...
  }
}
```

Baseline: just run similarity, sum over pairwise similarities of sentences. This shouldn't work too well.
Similarity measures:
* S-BERT?
* Arora?
