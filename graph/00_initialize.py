from sqlalchemy import create_engine
import graph_lib

if __name__ == "__main__":
  engine = create_engine("sqlite:///elife.db", future=True)
  graph_lib.Base.metadata.create_all(engine)
