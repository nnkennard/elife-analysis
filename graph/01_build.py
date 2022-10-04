import csv
import json

from sqlalchemy import create_engine, select, or_
from sqlalchemy.orm import Session

from graph_lib import Paper, Person, Authorship, Alias

def pubmed_row_generator():
  with open("pubmed_data.tsv", "r") as f:
    pubmed_reader = csv.DictReader(f, delimiter='\t')
    for row in pubmed_reader:
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
      paper = Paper(title=pubmed_row['title'])
      session.add(paper)
      session.flush()
      print(paper.id)
      authorships_and_aliases = []
      for i, author in enumerate(json.loads(pubmed_row['authors'])):
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

