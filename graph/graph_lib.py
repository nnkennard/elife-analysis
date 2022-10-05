from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, ForeignKey

Base = declarative_base()


class Paper(Base):
    __tablename__ = "paper"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    pm_id = Column(String)
    doi = Column(String)
    issn = Column(String)

    def __repr__(self):
        return f"Title:{self.title} Paper ID:{self.paper_id}"


class Person(Base):
    __tablename__ = "person"

    id = Column(Integer, primary_key=True)
    elife_id = Column(String)
    pm_author_id = Column(String)
    orcid = Column(String)

    def __repr__(self):
        return f"It's a person idk {self.id}"


class Authorship(Base):
    __tablename__ = "authorship"

    id = Column(Integer, primary_key=True)
    person_id = Column(String, ForeignKey("person.id"))
    paper_id = Column(String, ForeignKey("paper.id"))
    author_order = Column(Integer)


class Alias(Base):
    __tablename__ = "alias"

    id = Column(Integer, primary_key=True)
    person_id = Column(String, ForeignKey("person.id"))
    paper_id = Column(String, ForeignKey("paper.id"))
    first = Column(String)
    last = Column(String)
    initials = Column(String)
    email = Column(String)
    affiliation = Column(String)
