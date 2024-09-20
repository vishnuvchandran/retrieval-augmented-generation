import psycopg2
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase


load_dotenv()

def connect_db():
    connection_string = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
    return SQLDatabase.from_uri(connection_string)