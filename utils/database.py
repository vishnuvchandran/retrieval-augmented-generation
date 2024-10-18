import psycopg2
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
import mysql.connector
import json
import streamlit as st


load_dotenv()

DB_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT")
}

def get_db_connection():
    input_db_params = {
        "database": st.session_state.db_name,
        "user": st.session_state.db_user,
        "password": st.session_state.db_password,
        "host": st.session_state.host,
        "port": st.session_state.port
    }
    if st.session_state.database == "PostgreSQL":
        try:
            connection = psycopg2.connect(**input_db_params)
            return connection
        except psycopg2.Error as e:
            return False
    elif st.session_state.database == "MySQL":
        try:
            connection = mysql.connector.connect(**input_db_params)
            return connection
        except mysql.connector.Error as e:
            st.write(e)
            return False
    else:
        return False


def connect_db():
    if st.session_state.database == "PostgreSQL":
        connection_string = f"postgresql://{st.session_state.db_user}:{st.session_state.db_password}@{st.session_state.host}/{st.session_state.db_name}"
    elif st.session_state.database == "MySQL":
        connection_string = f"mysql+pymysql://{st.session_state.db_user}:{st.session_state.db_password}@{st.session_state.host}:{st.session_state.port}/{st.session_state.db_name}"
    else:
        return False
    # connection_string = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
    return SQLDatabase.from_uri(connection_string)
