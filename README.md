# elife-analysis

## Locating targets of criticism

I used [Stanza](https://stanfordnlp.github.io/stanza/) to annotate DISAPERE reviews with dependency parses, then trained a logistic regression classifier (using [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)) to classify sentences as 'some polarity' or 'no polarity' (not distinguishing between positive and negative polarity, for now).

Surprisingly, using dependency path features of length 2 resulted in a reasonable performance on the dev set of DISAPERE (86% accuracy). This classifier represented each sentence as a bag of dependency paths, of the form `(deprel_1_2, POS_2, deprel_2_3, POS_3)`, so notably, does not include any lexical information. This hopefully will guard against a severe drop in performance when switching to the biomedical domain. That said, these features are not great, and actually quite unsophisticated. There is more to be done, but here are the results with these naive features for now.

All examples in the readme are from [this review](https://openreview.net/forum?id=YTWGvpFOQD-&noteId=U5WeIre4ggR).

### Example featurization:


_Hence, I do not consider this as a new insight or a contribution._

```
('root', 'VB', 'advmod', 'RB'): 2, 
('root', 'VB', 'punct', ','): 1, 
('root', 'VB', 'nsubj', 'PRP'): 1, 
('root', 'VB', 'aux', 'VBP'): 1, 
('root', 'VB', 'obj', 'DT'): 1, 
('root', 'VB', 'obl', 'NN'): 1, 
('root', 'VB', 'punct', '.'): 1, 
('obl', 'NN', 'case', 'IN'): 1,
('obl', 'NN', 'det', 'DT'): 1, 
('obl', 'NN', 'amod', 'JJ'): 1,
('obl', 'NN', 'conj', 'NN'): 1,
('conj', 'NN', 'cc', 'CC'): 1,
('conj', 'NN', 'det', 'DT'): 1
```

You can use the [CoreNLP demo](corenlp.run) for a diagram of the parse represented above.

### Results of logistic regression classifier:

| Data split | Accuracy |
|------------|----------|
| Train      | 91.39%   |
| Dev        | 86.46%   |

### Results on unseen data

I applied this classifier to reviews from ICLR 2021 (all posted after any review in DISAPERE). Along with highlighting which sentences supposedly have or don't have some polarity, I extracted maximal noun phrases from the sentences that were classified as having some polarity.

[Results](https://nnkennard.github.io/elife-analysis/)

### Remaining action items:
* Distinguishing between positive and negative polarity sentences (without labels in the new domain)
  * Turney 2002 [Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews](https://aclanthology.org/P02-1053)
  * Zeng et al. 2020 [A Variational Approach to Unsupervised Sentiment Analysis](https://arxiv.org/abs/2008.09394)
* Determining which of the NPs are the targets of criticism, e.g.
  * _Hence, I do not consider **this** as a new insight or a contribution._
  * _**The presentation of results (Table 3)** is a bit strange._
* Linking targets of criticism to a NP that is more descriptive (something earlier in the coreference chain)
* Dealing with things that aren't going to end up being in a coreference chain. E.g. bolded phrases below are coreferent, but no major coref datasets do discourse deixis
  * _**The main focus and the message in the paper** is that the handcrafted features work better compared to learned features during training of NNs and having more training data results in better outcomes (i.e. a better privacy-utility trade-off)._)
  * _Starting with the latter, this is apparent from the noise formulation in DPSGD, where the noise is reduced via sampling probability, which decreases as the data size grows_
  * _Hence, I do not consider **this** as a new insight or a contribution._
* Improving classifier input features

