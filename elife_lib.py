class Node(object):

  def __init__(self, my_id, head, deprel, pos):
    self.my_id = my_id
    self.head = head
    self.deprel = deprel
    self.pos = pos
    self.children = []


def build_dep_tree(sentence):
  nodes = {0: Node(0, "ROOT", "ROOT", "ROOT")}
  for token in sentence.tokens:
    token_dict, = token.to_dict()
    my_id, head, deprel, pos, tok = tuple(
        [token_dict[key] for key in "id head deprel xpos text".split()])
    nodes[my_id] = Node(my_id, head, deprel, pos)
  for node_id, node in nodes.items():
    if not node_id:
      continue
    nodes[node.head].children.append(node)

  return nodes[0]


def extract_dep_paths(sentence, max_len=2):
  root = build_dep_tree(sentence)
  # Since I only want paths of length 2, just going up and down from each node
  stack = [root]
  features = []
  while True:
    if not stack:
      break
    curr_node = stack.pop(0)
    for child in curr_node.children:
      if curr_node.my_id:
        features.append(
            (curr_node.deprel, curr_node.pos, child.deprel, child.pos))
      stack.append(child)
  return features


def get_nsubj_subtree(sentence):
  dep_tree_root = build_dep_tree(sentence)
  actual_root, = dep_tree_root.children
  for child in actual_root.children:
    if child.deprel == 'nsubj':
      grandchild_indices = [x.my_id for x in child.children]
      if grandchild_indices:
        min_idx, max_idx = min(grandchild_indices), max(grandchild_indices)
        print(sentence.text)
        print()
        print(" ".join([x.text for x in sentence.tokens[min_idx:max_idx + 1]]))
        print("-" * 80 + "\n")


STANZA_PIPELINE = stanza.Pipeline("en", processors="tokenize")

def sentencize(review):
  annotated = STANZA_PIPELINE(review["review_text"])
  sentences = []
  for i, sentence in enumerate(annotated.sentences):
    sentences.append({"identifier": f'{review["review_id"]}_{i}',
    "text": sentence.text})
  return sentences


def sentencize_dir(data_dir):
  examples = {}
  for filename in glob.glob(f"{data_dir}/*.json"):
    with open(filename, 'r') as f:
      obj = json.load(f)
      for review in obj:
        examples[review['review_id']] = sentencize(review)
  return examples


