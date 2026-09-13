# app/db/session.py

# python script to create engine and establish session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base
import os
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(
    os.environ["DATABASE_URL"],
    pool_pre_ping=True,  # test each connection with a cheap ping before handing it out
    pool_recycle=280,  # proactively retire connections before Neon's 5-min suspend window
)
Sessionlocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
