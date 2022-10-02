import sqlite3


def main():

  con = sqlite3.connect("elife.db")
  cur = con.cursor()
  cur.execute("""CREATE TABLE paper(
    paper_id INTEGER PRIMARY KEY,
    source_type TEXT NOT NULL,
    elife_manuscript_id TEXT,
    pm_id TEXT,
    doi TEXT,
    issn TEXT,
    title TEXT NOT NULL,
    date_revised TEXT,
    date_completed TEXT,
    journal_pub_date TEXT
    )""")

  cur.execute("""CREATE TABLE person(
    person_id INTEGER PRIMARY KEY,
    elife_id TEXT,
    orcid TEXT
    )""")

  cur.execute("""CREATE TABLE alias(
    person_id INTEGER,
    paper_id INTEGER,

    last_name TEXT,
    first_name TEXT,
    initials TEXT,
    affiliation TEXT
    )""")

  cur.execute("""CREATE TABLE authorship(
    person_id INTEGER PRIMARY KEY,
    paper_id INTEGER,
    author_order INTEGER
    )""")

  cur.execute("""CREATE TABLE citation(
    citer_id INTEGER,
    citee_id INTEGER
    )""")

  con.commit()


if __name__ == "__main__":
  main()
