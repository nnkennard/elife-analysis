import csv
import sqlite3
"""Random notes

  * Different ways a paper can be IDed
    * PMID (PM)
    * DOI (PM)
    * ISSN (PM)



"""




def create_pubmed_alias(cursor, paper, info):
  pass
  # Check for scientist
  # Maybe create scientist
  # Create alias


def create_pubmed_citations(cursor, citer, pubmed_citation):
  pass
  # Check for citee
  # Maybe create citee
  # Create alias


PAPER_KEYS = "elife_manuscript_id pm_id doi issn".split()


def insert_or_create_paper(cursor, paper_row):
  key_string = " OR ".join([
      f"{key} == {value}" for key, value in paper_row.items()
      if key in PAPER_KEYS and value
  ])
  found_counter = {}
  maybe_matching_rows = cursor.execute(f"""
    SELECT * FROM paper WHERE {key_string}
  """).fetchall()
  assert len(maybe_matching_rows) in [0, 1]
  if maybe_matching_rows:
    (row,) = maybe_matching_rows
    return row[0]
  else:
    value_string_builder = {}
    for (key) in (
        "pm_id issn doi title date_revised date_completed journal_pub_date".
        split()):
      if key in paper_row:
        value_string_builder[key] = paper_row[key]
    sorted_keys = list(sorted(value_string_builder.keys()))
    key_string = ", ".join(["source_type", "paper_id"] + sorted_keys)
    value_string = f'"Pubmed", NULL,' + ", ".join(
        f'"{value_string_builder[key]}"' for key in sorted_keys)

    cursor.execute(
        f""" INSERT INTO paper({key_string}) VALUES ({value_string});""")
    (row,) = cursor.execute(f"""SELECT paper_id FROM paper ORDER BY paper_id
    DESC LIMIT 1""").fetchall()
    return row[0]


def process_pubmed_row(cursor, row):
  paper_id = insert_or_create_paper(cursor, row)

  # Check if paper is in Papers
  # If no, create paper
  # Create aliases
  # Alias is uniquely identified by order and paper
  # Create citations


def pubmed_row_generator():
  with open("pubmed.csv", "r") as f:
    pubmed_reader = csv.DictReader(f)
    for row in pubmed_reader:
      yield row


def main():

  con = sqlite3.connect("elife.db")
  cur = con.cursor()

  for pubmed_row in pubmed_row_generator():
    process_pubmed_row(cur, pubmed_row)
  con.commit()


if __name__ == "__main__":
  main()
