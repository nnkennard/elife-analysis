import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, MetaData


def main():

  engine = sqlalchemy.create_engine('sqlite:///elife.db')
  meta = MetaData()

  paper = Table(
      "paper",
      meta,
      Column("paper_id", Integer, primary_key=True),
      Column("source_type", String),
      Column("elife_manuscript_id", String),
      Column("pm_id", String),
      Column("doi", String),
      Column("issn", String),
      Column("title", String),
      Column("date_revised", String),
      Column("date_completed", String),
      Column("journal_pub_date", String),
  )

  person = Table(
      "person",
      meta,
      Column("person_id", Integer, primary_key=True),
      Column("elife_id", String),
      Column("orcid", String),
      Column("pm_author_id", String),
  ) 

  authorship = Table(
      "authorship",
      meta,
      Column("person_id", Integer, foreign_key="person.person_id"),
      Column("paper_id", Integer, foreign_key="paper.paper_id"),
      Column("author_order", Integer),
  )

  alias = Table(
      "alias",
      meta,
      Column("person_id", Integer, foreign_key="person.person_id"),
      Column("paper_id", Integer, foreign_key="paper.paper_id"),
      Column("first_name", String),
      Column("last_name", String),
      Column("affiliation", String),
      Column("initials", String),
  )

  citation = Table(
  "citation",
  meta,
  Column("citer_id", Integer, foreign_key="paper.paper_id"),
  Column("citee_id", Integer, foreign_key="paper.paper_id")
  )

  meta.create_all(engine)


if __name__ == "__main__":
  main()
