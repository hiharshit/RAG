# RAG Stack

A Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and Google's Gemini embeddings.

## Setup

1. Install dependencies with UV:
```bash
uv sync
```

2. Create a `.env` file in the root directory and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

3. Add your documents to the `docs/` directory as `.txt` files.

## Usage

### Ingestion Pipeline

Loads documents, splits them into chunks, and creates embeddings stored in ChromaDB:

```bash
uv run python rag-stack/01_ingestion_pipeline.py
```

This will:
- Load all `.txt` files from `docs/`
- Split documents into chunks
- Generate embeddings using Google's Gemini
- Store in `db/chroma_db/`

### Retrieval Pipeline

Query the vector database for relevant documents:

```bash
uv run python rag-stack/02_retrieval_pipeline.py
```

Enter your query when prompted. The system will return the most relevant document chunks.

## Project Structure

- `docs/` - Source documents (.txt files)
- `01_ingestion_pipeline.py` - Document ingestion and embedding
- `02_retrieval_pipeline.py` - Query retrieval
- `db/chroma_db/` - Vector database (ignored by git)

## Dependencies

- LangChain
- ChromaDB
- Google Generative AI (embeddings)
- python-dotenv
