import collections
import joblib
import elife_lib
import pandas as pd
import stanza

FULL_STANZA_PIPELINE = stanza.Pipeline('en',
            processors='tokenize,lemma,pos,depparse,constituency',)

polarity_classifier = joblib.load("ckpt/polarity_classifier.joblib")

def get_maximal_nps(tree):
    np_spans = []
    stack = [tree]
    while stack:
        this_tree = stack.pop(0)
        if this_tree.label == 'NP':
            np_spans.append(" ".join(this_tree.leaf_labels()))
        else:
            stack += this_tree.children
    return np_spans

def get_nps_with_polarity_table(text, split_sentences=True):
    dataframe_rows = []
    doc = FULL_STANZA_PIPELINE(text)
    examples = [collections.Counter(elife_lib.extract_dep_paths(sentence)) for sentence in doc.sentences]
    for i, (sentence, label) in enumerate(
        zip(doc.sentences,polarity_classifier.predict(examples))):
        row_builder = {
            "idx":i,
            "text":sentence.text,
            "polarity_label": label,
        }
        if label == "some_polarity":
            row_builder["np_list"] = ", ".join(get_all_nps(sentence.constituency))
        else:
            row_builder["np_list"] = ""
        dataframe_rows.append(row_builder)
    
    return pd.DataFrame.from_dict(dataframe_rows)
    
            

REVIEW ="""In general, it is not clear, at least theoretically, how, and when unsupervised data helps to generalization of nonlinear methods. In the literature there are important and elegant works exists that analyses the impact of usage of unlabelled data during training, however, (if I am not mistaken) all these analyses have been done for linear models. Authors analyse and shed some light to several aspects of using unlabelled data during training. They formalize their analyses based on expansion assumption. I think it can be restated as the similarity between members of the same classes is bounded by below. Intuitively such an assumption is quite reasonable. The authors use the term input consistency for defining a broad set of methods e.g. transformations of the image should be similar to each other, and they also couple their analysis using the expansion assumption with input consistency. In their view input consistency brings a local stability/generalization and expansion property brings global stability/generalization. This is again quite reasonable way of thinking because intuitively just forcing an input point to be close to transformed version of itself sounds a weak property for a good generalization performance. The authors supply quite a bit of theoretical novel material to support their intuition and analysis. Furthermore, they present some supportive experiments albeit not an extensive one.

Strong and weak points: a) Strong points: Please see above. b) The paper is quite dense, and the reader needs to be familiar with learning theory concepts. I wonder if authors would have focused on only one aspect of the problem which they are dealing. In the current version semi-supervised learning methods, unsupervised domain adaptation and unsupervised learning are covered. Every of them is a field by itself. I understand the desire of a unified and generic framework however I can imagine that there is a risk of diluting the message. c) Another understandably weak point is the experiments. I personally think that conducting an experimental study in the scope of this quite challenging however it will be nice see expansion property on a real dataset. Recommendation: Overall, I would like this paper to get published because (if I am not mistaken) paper develops an initial understanding extremely important field e.g. self-training/self-supervised learning.

Supporting arguments: a) I found the assumptions paper quite intuitive and necessary. Authors also supply population level guarantees for unsupervised learning. Moreover, they extend their work finite-sample guarantees by using margin concept and Lipschitz continuity. They extend their work to domain adaptation and semi-supervised learning. The novel material in the paper is extensive.

Questions: a) As I mentioned before I would like to expansion property on some real-world datasets. For example, can authors present some evidence of expansion property for a chosen deep neural network on a dataset (or multiple datasets) and quantify the expansion property based on some metric.
Improvement Suggestions: a) The paper is quite dense, and the reader needs to be familiar with learning theory concepts. I would recommend authors to decrease density of the paper and may be move some parts to the supplementary material.

Although I am quite positive about paper, I would like to see the discussion and comments. I am open to change my review to any direction if some new evidence/discussion/published work supplied.
"""

df = get_nps_with_polarity_table(REVIEW)