# Retrieval-Augmented Generation (RAG) Application

This project is a **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF and text documents, ask questions, and retrieve relevant data using vector similarity search. It leverages **ChromaDB** for embedding storage and search, **PostgreSQL** for dynamic SQL generation via text-to-SQL, and integrates with **Google Generative AI** for generating responses.

## Features
- **Document Upload**: Upload PDF and text files for storage and search.
- **Vector Similarity Search**: Retrieves context using **ChromaDB**.
- **Text-to-SQL**: Dynamically generates SQL queries using a Large Language Model (LLM) based on user prompts.
- **Conversational History**: Retrieves context based on conversation history for more coherent responses.
- **Flexible LLM Integration**: Supports **Google Generative AI** and can be expanded to other models like **OpenAI**.

## Development Environment
This project is developed using **Docker** on **WSL2**. It includes the following services:
- **App**: The core application running on Streamlit.
- **PostgreSQL**: The database used for SQL query generation and storage.
- **ChromaDB**: A vector database for storing embeddings and performing vector similarity search.
- **PgAdmin**: A web-based database management tool for PostgreSQL.

## Prerequisites
- Docker installed with WSL2 backend.
- API keys for **Google Generative AI**.
- Environment variables for database credentials.

## Installation and Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/vishnuvchandran/retrieval-augmented-generation.git
    cd retrieval-augmented-generation
    ```

2. **Set up environment variables**:  
   Create a `.env` file in the root directory with the following variables:
    ```bash
    POSTGRES_USER=your_postgres_user
    POSTGRES_PASSWORD=your_postgres_password
    POSTGRES_DB=your_postgres_db
    PGADMIN_EMAIL=your_pgadmin_email
    PGADMIN_PASSWORD=your_pgadmin_password
    GOOGLE_API_KEY=your_google_api_key
    ```

3. **Build and run the application**:
    Use `docker-compose` to build and start the application.
    ```bash
    docker-compose up --build
    ```

4. **Access the application**:
    - **Streamlit App**: Open [http://localhost:8501](http://localhost:8501).
    - **PgAdmin**: Access PgAdmin at [http://localhost:5050](http://localhost:5050) using the credentials from your `.env` file.

5. **Stopping the application**:
    ```bash
    docker-compose down
    ```

Enjoy using the RAG application! 🎉