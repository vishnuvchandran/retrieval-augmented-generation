from utils.vector_store import search_vectors, get_retriever
from utils.database import connect_db
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils.llm_selection import get_llm, get_embedding_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import FAISS
import re
from langchain.document_loaders import JSONLoader
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
import json


store = {}


def process_query(query: str):

    relevant_chunks = search_vectors(query, n_results=5)
    llm = get_llm('google')
    
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

    retriever = get_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": "session1"}},
    )["answer"]
    
    return response



def process_text_to_sql(query: str):
    embedding = get_embedding_model('google')
    llm = get_llm('google')
    pgdb = connect_db()
    documents = JSONLoader(file_path='./schema.jsonl', jq_schema='.', text_content=False, json_lines=True).load()
    db = FAISS.from_documents(documents=documents, embedding=embedding)
    
    retriever = db.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'lambda_mult': 1})
    matched_documents = retriever.get_relevant_documents(query=query)

    matched_tables = []

    for document in matched_documents:
        page_content = document.page_content
        page_content = json.loads(page_content)
        table_name = page_content['table_name']
        matched_tables.append(f'{table_name}')

    search_kwargs = {
        'k': 20
    }
    
    retriever = db.as_retriever(search_type='similarity', search_kwargs=search_kwargs)
    matched_columns = retriever.get_relevant_documents(query=query)

    matched_columns_filtered = []

    for i, column in enumerate(matched_columns):
        page_content = json.loads(column.page_content)
        matched_columns_filtered.append(page_content)
    
    matched_columns_cleaned = []
    
    for table in matched_columns_filtered:
        table_name = table['table_name']
        for column in table['columns']:
            column_name = column['name']
            data_type = column['type']
            matched_columns_cleaned.append(f'table_name={table_name}|column_name={column_name}|data_type={data_type}')
    
    matched_columns_cleaned = '\n'.join(matched_columns_cleaned)

    messages = []

    # template = "You are a SQL master expert capable of writing complex SQL queries in PostgreSQL."
    template = """You are a PostgreSQL expert capable of writing complex SQL queries. 
    Always use PostgreSQL-specific syntax and functions where appropriate. 
    Ensure all queries are optimized for PostgreSQL execution."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    messages.append(system_message_prompt)

    human_template = """Given the following inputs:
    USER_QUERY:
    --
    {query}
    --
    MATCHED_SCHEMA: 
    --
    {matched_schema}
    --
    Please construct a SQL query using the MATCHED_SCHEMA and the USER_QUERY provided above.

    IMPORTANT: Use ONLY the column names (column_name) mentioned in MATCHED_SCHEMA. DO NOT USE any other column names outside of this. 
    IMPORTANT: Associate column_name mentioned in MATCHED_SCHEMA only to the table_name specified under MATCHED_SCHEMA.
    NOTE: Use SQL 'AS' statement to assign a new name temporarily to a table column or even a table wherever needed. 
    """

    human_message = HumanMessagePromptTemplate.from_template(human_template)
    messages.append(human_message)

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    request = chat_prompt.format_prompt(query=query, matched_schema=matched_columns_cleaned).to_messages()
    
    response = llm.invoke(request)
    sql_query = '\n'.join(response.strip().split('\n')[1:-1])
    result = pgdb.run(sql_query)

    final_template = """
    Here is the result of your query:

    User Query:
    {user_query}

    Generated SQL Query:
    {sql_query}

    Query Result:
    {result}

    Instructions:
    1. DO NOT mention or describe the SQL query in your response.
    2. Focus on answering the user's question directly and clearly.
    3. If appropriate, provide a brief explanation or context for the answer.
    """


    final_response = llm.invoke(
        final_template.format(user_query=query, sql_query=sql_query, result=result)
    )
    
    return final_response
