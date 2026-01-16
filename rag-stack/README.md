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

### Semantic Chunking (Example)

Demonstrates semantic chunking using embeddings to split text based on meaning rather than fixed sizes:

```bash
uv run python rag-stack/05_semantic_chunking.py
```

This will:
- Use LangChain's SemanticChunker with Google embeddings
- Split sample text into semantically coherent chunks
- Display the resulting chunks with character counts
- Useful for improving retrieval quality by keeping related information together

### Agentic Chunking (Example)

Demonstrates AI-powered chunking where an LLM intelligently splits text based on content:

```bash
uv run python rag-stack/06_agentic_chunking.py
```

This will:
- Use an LLM to analyze and split text at logical boundaries
- Generate chunks around 200 characters based on topic changes
- Use a custom delimiter (`<<<SPLIT>>>`) to mark split points
- Display the resulting chunks with character counts
- Shows how AI can understand context for better chunking decisions

## Project Structure

- `docs/` - Source documents (.txt files)
- `01_ingestion_pipeline.py` - Document ingestion and embedding
- `02_retrieval_pipeline.py` - Query retrieval
- `03_answer_generation.py` - Answer generation using LLM
- `04_history_aware_generation.py` - Chat interface with conversation history
- `05_semantic_chunking.py` - Semantic chunking example
- `06_agentic_chunking.py` - AI-powered chunking example
- `db/chroma_db/` - Vector database (ignored by git)

## Dependencies

- LangChain
- ChromaDB
- Google Generative AI (embeddings)
- python-dotenv
