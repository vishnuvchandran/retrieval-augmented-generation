import streamlit as st
from utils.document_loader import load_document_from_uploadedfile, load_document_from_url
from utils.text_splitter import split_text
from utils.vector_store import store
from utils.query_processing import process_query, process_text_to_sql
from utils.schema_index import SchemaIndexer


# Function to load the SVG file content
def load_svg(file_path):
    with open(file_path, "r") as f:
        return f.read()


def submit_url():
    st.session_state.web_url = st.session_state.url
    st.session_state.url = ""


def main():
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="imgs/pits.svg",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "About": """
                ## Retrieval-Augmented Generation (RAG) Application
                ### Powered using Google Generative AI

                This system allows users to upload PDF and text documents, ask questions, 
                and retrieve relevant data using vector similarity search. 
                It leverages ChromaDB for embedding storage and search, PostgreSQL for dynamic SQL generation via text-to-SQL, 
                and integrates with Google Generative AI for generating responses.
            """
        }
    )

    st.markdown("""
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            .stAppDeployButton {display:none;}
        </style>
    """, unsafe_allow_html=True)

    # Streamlit Title
    st.title("Hello! How can I assist you today?")

    # Load and display sidebar SVG image
    img_path = "imgs/pits.svg"
    img_svg = load_svg(img_path)

    if img_svg:
        st.sidebar.markdown(
            f'{img_svg}',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    # Application type selection dropdown
    app_type = st.sidebar.radio(
        "Select Mode:",
        ("Document Assistant", "Data Explorer"),
        index=0
    )

    if app_type == "Document Assistant":
        st.sidebar.markdown("""
        **Document Assistant** helps you search through your uploaded documents 
        and retrieve relevant information by asking natural language questions.
        """)
    elif app_type == "Data Explorer":
        st.sidebar.markdown("""
        **Data Explorer** allows you to ask questions in natural language, 
        and it will generate SQL queries to retrieve data from your database.
        """)

    st.sidebar.markdown("---")

    
    # Sidebar for document upload
    if app_type == "Document Assistant":
        with st.sidebar:
            st.title("Document Upload")
            uploaded_file = st.file_uploader("Choose a document", type=["txt", "pdf"])
            
            if uploaded_file is not None:
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        # Document processing pipeline
                        doc = load_document_from_uploadedfile(uploaded_file)
                        chunks = split_text(doc)
                        store(chunks)
                    st.success("Document processed successfully!")

        st.sidebar.markdown("---")

    if 'db_connect' not in st.session_state:
        st.session_state.db_connect = False

    if 'schema' not in st.session_state:
        st.session_state.schema = None

    if st.session_state.get('disconnect'):
        st.session_state['database'] = 'PostgreSQL'
        st.session_state['host'] = 'localhost'
        st.session_state['port'] = '5432'
        st.session_state['db_name'] = ''
        st.session_state['db_user'] = ''
        st.session_state['db_password'] = ''

    # Capture database connection details
    if app_type == "Data Explorer":
        with st.sidebar:
            st.title("Database Details")
            db_type = st.selectbox("Database Type:", ["PostgreSQL", "MySQL"], key='database')
            host = st.text_input("Host:", value="localhost", key='host')
            port = st.text_input("Port:", value="5432" if db_type == "PostgreSQL" else "3306", key='port')
            database = st.text_input("Database Name:", key='db_name')
            user = st.text_input("User:", key='db_user')
            password = st.text_input("Password:", type="password", key='db_password')

            # Show connection options based on connection state
            if st.session_state.db_connect:
                if st.button("Disconnect", key='disconnect'):
                    with st.spinner("Disconnecting from the database..."):
                        st.write('Disconnecting')
                    st.info("Disconnected successfully!")
                    st.session_state.db_connect = False
                    st.rerun()
            else:
                if st.button("Connect to Database"):
                    with st.spinner("Connecting to the database..."):
                        schema_indexer = SchemaIndexer()
                        if not schema_indexer.sqldb:
                            st.error("Failed to connect to the database. Please check your connection settings.")
                        else:
                            st.session_state.schema = schema_indexer
                            st.success("Connected successfully!")
                            st.session_state.db_connect = True
                            st.rerun()
            
                

    if "web_url" not in st.session_state:
        st.session_state.web_url = ""

    # Sidebar for Crawl the wepage and get the text content
    if app_type == "Document Assistant":
        with st.sidebar:
            st.title("Load Webpage Content")
            st.text_input("Enter the URL:", key="url", on_change=submit_url)
            url = st.session_state.web_url

            if url:
                with st.spinner("Processing..."):
                    doc = load_document_from_url(url)
                    chunks = split_text(doc)
                    store(chunks)
                    st.session_state.web_url = ""
                st.success("Document processed successfully!")


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the query and generate response
        with st.spinner("Generating response..."):
            if app_type == "Document Assistant":
                response = process_query(prompt)
            elif app_type == "Data Explorer":
                if st.session_state.db_connect:
                    result = process_text_to_sql(prompt, st.session_state.schema)
                    if result:
                        response = result
                    else:
                        response = ''
                        st.error("Database connection error.")

                else:
                    response = ''
                    st.error("No Database connection.")
            else:
                response = process_query(prompt)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()