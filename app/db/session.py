# app/db/connection.py

# python script to create engine and establish session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base
import os

engine = create_engine(os.environ["DATABASE_URL"])
Sessionlocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
