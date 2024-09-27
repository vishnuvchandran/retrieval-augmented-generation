import psycopg2
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
import json


load_dotenv()

DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT")
}

def get_db_connection():
    return psycopg2.connect(**DB_PARAMS)


def connect_db():
    connection_string = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
    return SQLDatabase.from_uri(connection_string)
