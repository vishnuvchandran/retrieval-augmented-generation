import json
from langchain.schema import Document
from langchain.vectorstores import FAISS
from utils.database import get_db_connection
from utils.llm_selection import get_embedding_model
import streamlit as st

class SchemaIndexer:
    def __init__(self):
        self.sqldb = get_db_connection()
        if not self.sqldb:
            return None
        self.embedding_model = get_embedding_model('google')
        self.db = self._create_index()


    def _fetch_schema_from_db(self):
        cursor = self.sqldb.cursor()
        schema_query = """
        SELECT 
            t.table_name,
            json_agg(json_build_object(
                'name', c.column_name,
                'type', c.data_type
            ))::text AS columns
        FROM 
            information_schema.tables t
        JOIN 
            information_schema.columns c ON t.table_name = c.table_name
        WHERE 
            t.table_schema = 'public'
        GROUP BY 
            t.table_name
        """
        cursor.execute(schema_query)
        schema = cursor.fetchall()
        cursor.close()
        
        documents = []
        for table in schema:
            doc = Document(
                page_content=json.dumps({
                    "table_name": table[0],
                    "columns": json.loads(table[1])
                }),
                metadata={"table": table[0]}
            )
            documents.append(doc)
        
        return documents


    def _create_index(self):
        documents = self._fetch_schema_from_db()
        return FAISS.from_documents(documents=documents, embedding=self.embedding_model)


    def get_index(self):
        return self.db