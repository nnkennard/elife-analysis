import csv
import sqlite3


# TODO(dss): what is the paper identifier in pubmed reference objects?
class Paper(object):
  IDENTIFIERS = "elife_manuscript_id pm_id doi issn".split()
  USER_KEYS = (
      IDENTIFIERS +
      "title date_revised date_completed journal_pub_date journal".split())
  # ALL_KEYS = "source_type paper_id".split() + USER_KEYS


# TODO(dss): do we have any other person identifiers?
class Person(object):
  IDENTIFIERS = "elife_id orcid".split()
  USER_KEYS = IDENTIFIERS


def check_for_any_keys(cursor, row, table, identifiers):
  key_string = " OR ".join([
      f"{key} == {row[key]}" for key in identifiers if key in row and row[key]
  ])
  found_counter = {}
  maybe_matching_rows = cursor.execute(f"""
    SELECT * FROM {table} WHERE {key_string}
  """).fetchall()
  assert len(maybe_matching_rows) in [0, 1]
  return maybe_matching_rows


def insert_or_update_paper(cursor, paper_row):
  maybe_matching_rows = check_for_any_keys(cursor, paper_row, "paper",
                                           Paper.IDENTIFIERS)
  if maybe_matching_rows:
    # Check for updates and/or conflicts
    (row,) = maybe_matching_rows
    return row[0]
  else:
    builder = {
        key: paper_row[key] for key in Paper.USER_KEYS if key in paper_row
    }
    key_string = ", ".join(["source_type", "paper_id"] + sorted(builder.keys()))
    value_string = f'"Pubmed", NULL,' + ", ".join(
        f'"{builder[key]}"' for key in sorted(builder.keys()))
    k = cursor.execute(
        f""" INSERT INTO paper({key_string}) VALUES ({value_string});""")
    (row,) = cursor.execute(f"""SELECT last_insert_rowid()""").fetchall()
    return row[0]


def insert_or_update_person(cursor, person_row):
  maybe_matching_rows = check_for_any_keys(cursor, person_row, "person",
                                           Person.IDENTIFIERS)
  if maybe_matching_rows:
    (row,) = maybe_matching_rows
    # Check for updates and/or conflicts
    return row[0]
  else:
    builder = {
        key: person_row[key] for key in Person.USER_KEYS if key in person_row
    }
    key_string = ", ".join(["person_id"] + sorted(builder.keys()))
    value_string = f"NULL," + ", ".join(
        f'"{builder[key]}"' for key in sorted(builder.keys()))
    k = cursor.execute(
        f""" INSERT INTO paper({key_string}) VALUES ({value_string});""")
    (row,) = cursor.execute(f"""SELECT last_insert_rowid()""").fetchall()
    return row[0]


def process_pubmed_citation(cursor, row, citer_id):
  citee_id = insert_or_update_paper(row)
  # INSERT INTO citation VALUES citer_id, citee_id, PubMed


def insert_or_update_alias(cursor, row, paper_id, author_i):
  person_id = insert_or_update_person(row)
  # INSERT INTO authorship VALUES paper_id, person_id, author_order
  # INSERT INTO alias VALUES paper_id, person_id, first_name, last_name, affiliation
  # ^ These should not lead to conflicts because this is the first time this pair is in here


def process_pubmed_row(cursor, row):
  paper_id = insert_or_update_paper(cursor, row)
  # for author_i, author in enumerate(row['authors']):
  #  insert_or_update_alias(cursor, row, paper_id, author_i)
  # for reference in row['references']:
  # process_pubmed_citations(reference)


def pubmed_row_generator():
  with open("pubmed.csv", "r") as f:
    pubmed_reader = csv.DictReader(f)
    for row in pubmed_reader:
      # TODO(dss): before yielding, modify the keys to match the ones in Paper
      # and Person keys. Row should be in the format:
      # { key1: value1,
      #   key2: value2,
      #   ...
      #   "authors": [author_dict_1, author_dict_2, ...]
      #   "references": [reference_dict_1, reference_dict_2, ...]
      # }

      yield row


def main():

  con = sqlite3.connect("elife.db")
  cur = con.cursor()

  for pubmed_row in pubmed_row_generator():
    process_pubmed_row(cur, pubmed_row)

  # TODO(dss): Share example eLife data with Neha to start process_elife_row

  con.commit()


if __name__ == "__main__":
  main()
