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

### Answer Generation

Generate answers based on retrieved documents using a language model:

```bash
uv run python rag-stack/03_answer_generation.py
```

Enter your query when prompted. The system will retrieve relevant documents and generate an answer using Google's Gemini.

### History-Aware Generation

Interactive chat with conversation context that maintains history across multiple queries:

```bash
uv run python rag-stack/04_history_aware_generation.py
```

This will:
- Start an interactive chat session
- Maintain conversation history for context
- Rewrite follow-up questions to be standalone for better retrieval
- Generate answers based on retrieved documents and conversation context
- Type 'quit' to exit

## Project Structure

- `docs/` - Source documents (.txt files)
- `01_ingestion_pipeline.py` - Document ingestion and embedding
- `02_retrieval_pipeline.py` - Query retrieval
- `03_answer_generation.py` - Answer generation using LLM
- `04_history_aware_generation.py` - Chat interface with conversation history
- `db/chroma_db/` - Vector database (ignored by git)

## Dependencies

- LangChain
- ChromaDB
- Google Generative AI (embeddings)
- python-dotenv
