import csv

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from graph_lib import Paper

def pubmed_row_generator():
  with open("pubmed_data.tsv", "r") as f:
    pubmed_reader = csv.DictReader(f, delimiter='\t')
    for row in pubmed_reader:
      yield row


def check_for_author(author_row):
  pass

def main():

  engine = create_engine("sqlite:///elife.db", future=True)

  with Session(engine) as session:
    for pubmed_row in pubmed_row_generator():
      paper = Paper(title=pubmed_row['title'])
      session.add(paper)
      session.flush()
      print(paper.id)
      for author in pubmed_row['authors']:
        pass
        # Try to find the author
        # If not, create the author

    session.commit()


if __name__ == "__main__":
  main()

