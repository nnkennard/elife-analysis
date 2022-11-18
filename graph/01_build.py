import csv
import json

from sqlalchemy import create_engine, select, or_
from sqlalchemy.orm import Session

from graph_lib import Paper, Person, Authorship, Alias

from google.cloud import bigquery
from google.cloud import storage


# BQ_CLIENT = bigquery.Client()
BQ_PUBMED_DATA = bigquery.TableReference.from_string(table_id="gse-nero-dmcfarla-mimir.PubMed.PubMed_Baseline")
BQ_PUBMED_ITERATOR = BQ_CLIENT.list_rows(BQ_PUBMED_DATA, page_size = 100).to_dataframe_iterable()

def pubmed_row_generator():

  pubmed_cols = ['PMID', 'ArticleTitle', 
                 'DateCompleted','DateRevised', 
                 'JournalPubDate', 'ISSN', 'DOI',  
                 'AuthorList', 'ReferenceList']

  new_cols = ['pmid', 'title', 'date_completed', 
              'date_revised', 'journal_pub_date', 
              'issn', 'doi', 'authors', 'references']

  col_map = dict(zip(pubmed_cols,new_cols))

  new_auth_keys = ['last','first','affiliation', 'orcid', 'email']

  for df in BQ_PUBMED_ITERATOR:
    # select and rename subset of cols
    df = df[pubmed_cols]
    df.rename(columns=col_map, inplace=True)
      
    # select and rename subset of auth dict items
    for index, row in df.iterrows():
      
      authors = []
      for author_dct in row['authors']:
        author_dct.pop("Initials", None)
        author_dct.pop("Order", None)

        # Grab email if poss
        if author_dct["Affiliation"] is not None:
          for tkn in author_dct['Affiliation'].split():
            if "@" in tkn: 
              author_dct['email'] = tkn
            else: 
              author_dct['email'] = None
        else: 
          author_dct['email'] = None

        # rekey dict
        for old_k, new_k in zip(author_dct.keys(), new_auth_keys):
          author_dct[new_k] = author_dct.pop(old_k)

        authors.append(author_dct)
      row['authors'] = authors
      yield row


def get_or_create_author_id(author_row, session):
    # This is really messy, maybe fix later
    # Also not efficient but I can't do much better rn
    match_ids = []
    if "orcid" in author_row:
        match_ids += [
            i.id
            for i in session.execute(
                select(Person.id).where(Person.orcid == author_row["orcid"])
            ).fetchall()
        ]
    if "pm_author_id" in author_row:
        match_ids += [
            i.id
            for i in session.execute(
                select(Person.id).where(
                    Person.pm_author_id == author_row["pm_author_id"]
                )
            ).fetchall()
        ]

    assert len(set(match_ids)) in [0, 1]
    if match_ids:
        return match_ids[0]
    else:
        author = Person(
            orcid=author_row.get("orcid", None),
            pm_author_id=author_row.get("pm_author_id", None),
        )
        session.add(author)
        session.flush()
        return author.id


def main():

    engine = create_engine("sqlite:///elife.db", future=True)

    with Session(engine) as session:
        for pubmed_row in pubmed_row_generator():
            # Create a row for this paper in the paper table
            paper = Paper(title=pubmed_row["title"])
            session.add(paper)
            session.flush()  # This allows us to access paper.id later

            # Collect authorship and alias informationi
            authorships_and_aliases = []
            for i, author in enumerate(json.loads(pubmed_row["authors"])):
                # The author may already exist as a person in the person table
                author_id = get_or_create_author_id(author, session)
                authorships_and_aliases.append(
                    Authorship(person_id=author_id, paper_id=paper.id, author_order=i)
                )
                authorships_and_aliases.append(
                    Alias(
                        person_id=author_id,
                        paper_id=paper.id,
                        first=author.get("first", None),
                        last=author.get("last", None),
                    )
                )
            session.add_all(authorships_and_aliases)
            session.commit()


if __name__ == "__main__":
    main()
