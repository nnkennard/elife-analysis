"""Random notes

  * Different ways a paper can be IDed
    * PMID (PM)
    * DOI (PM)
    * ISSN (PM)



"""


def create_tables():
  # Paper
  # Scientist
  # Authorship
  # Alias

def locate_paper(doi=None, pmid=None, ):
  # SELECT where
  # Return Paper or None

def create_pubmed_paper(cursor, row):
  # INSERT INTO

def create_pubmed_alias(cursor, paper, info):
  # Check for scientist
  # Maybe create scientist
  # Create alias

def create_pubmed_citations(cursor, citer, pubmed_citation):
  # Check for citee
  # Maybe create citee
  # Create alias


def process_pubmed_row(cursor, row):
  # Check if paper is in Papers
  # If no, create paper
  # Create aliases
    # Alias is uniquely identified by order and paper
  # Create citations

def main():

  create_tables()

  pass


if __name__ == "__main__":
  main()

