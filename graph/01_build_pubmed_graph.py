import csv
import sqlite3


class Table(object):

  def __init__(
      self,
      name,
      non_identifier_user_keys,
      primary_key=None,
      foreign_keys=[],
      identifiers=[],
  ):
    self.name = name
    self.primary_key = primary_key
    self.identifiers = identifiers
    self.foreign_keys = foreign_keys
    self.user_keys = identifiers + non_identifier_user_keys


PAPER = Table(
    "paper",
    "title date_revised date_completed journal_pub_date journal".split(),
    primary_key="paper_id",
    identifiers="elife_manuscript_id pm_id doi issn".split(),
)

PERSON = Table("person", [],
               primary_key="person_id",
               identifiers="elife_id orcid".split())

AUTHORSHIP = Table("authorship", ["author_order"],
                   foreign_keys="paper_id person_id".split())

ALIAS = Table(
    "alias",
    "last_name first_name initials affiliation".split(),
    foreign_keys="paper_id person_id".split(),
)


def check_for_any_keys(cursor, row, table):
  key_string = " OR ".join([
      f"{key} == {row[key]}" for key in table.IDENTIFIERS
      if key in row and row[key]
  ])
  maybe_matching_rows = cursor.execute(
      f""" SELECT * FROM {table.NAME} WHERE {key_string}
  """).fetchall()
  assert len(maybe_matching_rows) in [0, 1]
  return maybe_matching_rows


def check_for_key_conflicts(existing_table_row, new_row):
  print(existing_table_row.keys(), new_row.keys())
  # TODO(nnk): Clean this up with feedback from actual data
  for key, value in row.items():
    pass


def insert_row(table, keyword_map, cursor):

  sorted_keys, sorted_values = zip(*[(key, keyword_map[key])
                                     for key in sorted(keyword_map)
                                     if keyword_map[key]])

  key_string = f"{table.PRIMARY_KEY}, " + ", ".join(sorted_keys)
  value_string = "NULL," + ", ".join([f'"{v}"' for v in sorted_values])
  print(f""" INSERT INTO {table.NAME}({key_string}) VALUES ({value_string});""")
  k = cursor.execute(
      f""" INSERT INTO {table.NAME}({key_string}) VALUES ({value_string});""")
  (row,) = cursor.execute(f"""SELECT last_insert_rowid()""").fetchall()
  return row[0]


def insert_or_update_paper(cursor, paper_row):
  maybe_matching_rows = check_for_any_keys(cursor, paper_row, Paper)
  if maybe_matching_rows:
    # Check for updates and/or conflicts
    (row,) = maybe_matching_rows
    return row[0]
  else:
    builder = {
        key: paper_row[key] for key in Paper.USER_KEYS if key in paper_row
    }
    builder["source_type"] = "PubMed"
    new_row_id = insert_row(Paper, builder, cursor)
    return new_row_id


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
    new_row_id = insert_row(Person, builder, cursor)
    return new_row_id


def process_pubmed_citation(cursor, row, citer_id):
  citee_id = insert_or_update_paper(row)
  cursor.execute(
      f""" INSERT INTO citation(citer_id, citee_id, source_type) VALUES
      ({citer_id}, {citee_id}, "PubMed");""")


def insert_or_update_alias(cursor, row, paper_id, author_order):
  person_id = insert_or_update_person(row)
  insert_row(
      Alias,
      {
          "paper_id": paper_id,
          "person_id": person_id,
          "author_order": str(author_order),
      },
      cursor,
  )
  insert_row(
      Alias,
      {
          "paper_id": paper_id,
          "person_id": person_id,
          "author_order": str(author_order),
      },
      cursor,
  )

  cursor.execute(
      f""" INSERT INTO alias(paper_id, person_id, author_order) VALUES
      ({paper_id}, {person_id}, "{author_order}");""")
  # INSERT INTO alias VALUES paper_id, person_id, first_name, last_name, affiliation
  # ^ These should not lead to conflicts because this is the first time this pair is in here


def process_pubmed_row(cursor, row):
  paper_id = insert_or_update_paper(cursor, row)
  # for author_i, author in enumerate(row['authors']):
  #   insert_or_update_alias(cursor, row, paper_id, author_i)
  # for reference in row['references']:
  #   process_pubmed_citations(reference, paper_id)


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
