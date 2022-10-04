from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class Paper(Base):
  __tablename__ = "paper"

  id = Column(Integer, primary_key=True)
  title = Column(String)
  pm_id = Column(String)
  doi = Column(String)
  issn = Column(String)

  def __repr__(self):
    return f'Title:{self.title} Paper ID:{self.paper_id}'
