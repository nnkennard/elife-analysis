import csv
import sqlalchemy as db



def process_pubmed_row(conn, row, paper):
  print(row)

  conn.execute(paper.insert(), [row]) # TODO(nnk): can i just pass the row?
  print(paper.id)
  #TODO (nnk): how to get pk of last insert
  # for author_i, author in enumerate(row['authors']):
  #   insert_or_update_alias(conn, row, paper_id, author_i)
  # for reference in row['references']:
  #   process_pubmed_citations(conn, reference, paper_id)


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

  engine = db.create_engine('sqlite:///elife.db')
  conn = engine.connect()
  metadata = db.MetaData()
  print(metadata)
  paper = db.Table('paper', metadata, autoload=True, autoload_with=engine)
  query = db.select([paper])
  result_proxy = conn.execute(query)
  result_set = result_proxy.fetchall()
  print(result_set)

  for pubmed_row in pubmed_row_generator():
    process_pubmed_row(conn, pubmed_row, paper)

  # TODO(dss): Share example eLife data with Neha to start process_elife_row


if __name__ == "__main__":
  main()
