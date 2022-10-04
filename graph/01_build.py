import csv
import json

from sqlalchemy import create_engine, select, or_
from sqlalchemy.orm import Session

from graph_lib import Paper, Person, Authorship, Alias

def pubmed_row_generator():
  with open("pubmed_data.tsv", "r") as f:
    pubmed_reader = csv.DictReader(f, delimiter='\t')
    for row in pubmed_reader:
			# TODO(dss): before yielding, modify the keys to match the ones in paper
      # and person keys. Row should be in the format:
      # { key1: value1,
      #   key2: value2,
      #   ...
      #   "authors": [author_dict_1, author_dict_2, ...]
      #   "references": [reference_dict_1, reference_dict_2, ...]
      # }
			# Author dicts should be in the format
			# { key1: value1,... } etc., but if there is no value provided for a
			# field e.g. ORCID, just don't include the key in the dict.

      yield row

def get_or_create_author_id(author_row, session):
  # This is really messy, maybe fix later
  # Also not efficient but I can't do much better rn
  match_ids = []
  if 'orcid' in author_row:
    match_ids += [i.id for i in session.execute(select(Person.id).where(
    Person.orcid == author_row['orcid'])).fetchall()]
  if 'pm_author_id' in author_row:
    match_ids += [i.id for i in session.execute(select(Person.id).where(
    Person.pm_author_id == author_row['pm_author_id'])).fetchall()]

  assert len(set(match_ids)) in [0,1]
  if match_ids:
    return match_ids[0]
  else:
    author = Person( orcid=author_row.get('orcid',
    None), pm_author_id=author_row.get('pm_author_id', None))
    session.add(author)
    session.flush()
    return author.id

def main():

  engine = create_engine("sqlite:///elife.db", future=True)

  with Session(engine) as session:
    for pubmed_row in pubmed_row_generator():
			# Create a row for this paper in the paper table
      paper = Paper(title=pubmed_row['title'])
      session.add(paper)
      session.flush() # This allows us to access paper.id later

			# Collect authorship and alias informationi
      authorships_and_aliases = []
      for i, author in enumerate(json.loads(pubmed_row['authors'])):
				# The author may already exist as a person in the person table
        author_id = get_or_create_author_id(author, session)
        authorships_and_aliases.append(
        Authorship(person_id=author_id, paper_id=paper.id,
                               author_order=i))
        authorships_and_aliases.append(
        Alias(person_id=author_id, paper_id=paper.id,
          first=author.get('first', None), last=author.get('last', None)))
      session.add_all(authorships_and_aliases)
      session.commit()


if __name__ == "__main__":
  main()

