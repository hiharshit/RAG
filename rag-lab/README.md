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
uv run python rag-lab/01_ingestion_pipeline.py
```

This will:

- Load all `.txt` files from `docs/`
- Split documents into chunks
- Generate embeddings using Google's Gemini
- Store in `db/chroma_db/`

### Retrieval Pipeline

Query the vector database for relevant documents:

```bash
uv run python rag-lab/02_retrieval_pipeline.py
```

Enter your query when prompted. The system will return the most relevant document chunks.

### Answer Generation

Generate answers based on retrieved documents using a language model:

```bash
uv run python rag-lab/03_answer_generation.py
```

Enter your query when prompted. The system will retrieve relevant documents and generate an answer using Google's Gemini.

### History-Aware Generation

Interactive chat with conversation context that maintains history across multiple queries:

```bash
uv run python rag-lab/04_history_aware_generation.py
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
uv run python rag-lab/05_semantic_chunking.py
```

This will:

- Use LangChain's SemanticChunker with Google embeddings
- Split sample text into semantically coherent chunks
- Display the resulting chunks with character counts
- Useful for improving retrieval quality by keeping related information together

### Agentic Chunking (Example)

Demonstrates AI-powered chunking where an LLM intelligently splits text based on content:

```bash
uv run python rag-lab/06_agentic_chunking.py
```

This will:

- Use an LLM to analyze and split text at logical boundaries
- Generate chunks around 200 characters based on topic changes
- Use a custom delimiter (`<<<SPLIT>>>`) to mark split points
- Display the resulting chunks with character counts
- Shows how AI can understand context for better chunking decisions

### Multi-Modal RAG (Jupyter Notebook)

Advanced RAG pipeline that handles documents with text, images, and tables:

```bash
jupyter notebook rag-lab/07_multi_modal_rag.ipynb
```

This notebook demonstrates:

- PDF partitioning using unstructured library (extracts text, tables, and images)
- Semantic chunking based on document structure
- AI-powered summarization for mixed content (text + tables + images)
- Vector store creation with ChromaDB
- Multi-modal query processing that considers all content types

**Note:** Requires system dependencies for PDF processing:

- Poppler (for PDF parsing)
- Tesseract (for OCR on images)
- libmagic (for file type detection)

Install with:

- **Linux:** `apt-get install poppler-utils tesseract-ocr libmagic-dev`
- **macOS:** `brew install poppler tesseract libmagic`
- **Windows:** Download from respective project websites (see notebook for links)

### Retrieval Methods (Example)

Demonstrates different retrieval strategies from the vector database:

```bash
uv run python rag-lab/08_retrieval_methods.py
```

This will:

- Show basic similarity search (top k most similar documents)
- Demonstrate similarity with score threshold (optional, uncomment to use)
- Show Maximum Marginal Relevance (MMR) for diverse results (optional, uncomment to use)
- Compare how different retrieval methods affect the results

**Retrieval methods included:**

1. **Similarity Search** - Returns top k most similar documents
2. **Similarity with Score Threshold** - Only returns documents above a certain similarity score
3. **Maximum Marginal Relevance (MMR)** - Balances relevance and diversity to avoid redundant results

### Multi-Query Retrieval (Example)

Demonstrates multi-query retrieval where an LLM generates multiple variations of a query to improve retrieval quality:

```bash
uv run python rag-lab/09_multi_query_retrieval.py
```

This will:

- Use an LLM to generate 3 different variations of the original query
- Retrieve documents for each query variation
- Store all results for potential Reciprocal Rank Fusion (RRF)
- Display results from each query variation side by side

**Benefits:**

- Improves retrieval by approaching the same question from different angles
- Helps find relevant documents that might not match the original query exactly
- Can be combined with RRF for better result ranking

### Reciprocal Rank Fusion (Example)

Demonstrates Reciprocal Rank Fusion (RRF) to combine and rank results from multiple query retrievals:

```bash
uv run python rag-lab/10_reciprocal_rank_fusion.py
```

This will:

- Generate multiple query variations using an LLM
- Retrieve documents for each query variation
- Apply RRF algorithm to fuse and rank all results together
- Display final ranking with RRF scores
- Show detailed scoring calculation with verbose output

**How RRF works:**

- Scores documents based on their position across multiple retrieval results
- Formula: `score = Σ 1/(k + position)` where k is typically 60
- Documents appearing in multiple queries get boosted scores
- Higher positions contribute more to the final score
- Provides a balanced fusion without depending on similarity scores

**Benefits:**

- Combines results from multiple retrieval strategies intelligently
- Doesn't require similarity scores to be on the same scale
- Proven effective in many retrieval and ranking tasks

### Hybrid Search (Jupyter Notebook)

Demonstrates hybrid search combining semantic (vector) and keyword (BM25) retrieval methods:

```bash
jupyter notebook rag-lab/11_hybrid_search.ipynb
```

This notebook demonstrates:

- Vector retriever for semantic search (understands meaning and concepts)
- BM25 retriever for keyword search (finds exact term matches)
- Ensemble retriever to combine both approaches with weighted fusion
- Example queries showing how hybrid search outperforms either method alone

**Benefits of hybrid search:**

- Combines semantic understanding with exact keyword matching
- Handles queries with mixed semantic and specific terms better
- Weights can be adjusted (e.g., 0.7 vector + 0.3 keyword)
- Often provides more accurate results than either approach alone

**Note:** Requires `langchain-classic` and `rank_bm25` packages if not already installed:

```bash
uv add langchain-classic rank_bm25
```

### Reranker (Jupyter Notebook)

Demonstrates using Cohere's Rerank model to improve retrieval results by reordering documents:

```bash
jupyter notebook rag-lab/12_reranker.ipynb
```

This notebook demonstrates:

- Hybrid search (vector + BM25) to retrieve initial candidate documents
- Cohere Rerank model to intelligently reorder results based on query relevance
- Comparison of results before and after reranking
- How reranking moves the most contextually relevant chunks to the top

**How reranking works:**

1. Retrieve more documents than needed from hybrid search (e.g., top 25)
2. Pass retrieved documents and query to the reranker
3. Reranker scores each document based on its relevance to the query
4. Returns top N documents with improved ordering

**Benefits:**

- Significantly improves result quality and relevance
- Handles noisy retrieval results effectively
- Specialized models outperform generic similarity scoring
- Critical for production RAG systems

**Note:** Requires `langchain-cohere` package and a Cohere API key:

```bash
uv add langchain-cohere
```

Add to `.env`:

```
COHERE_API_KEY=your_cohere_api_key_here
```

## Project Structure

- `docs/` - Source documents (.txt and .pdf files)
- `01_ingestion_pipeline.py` - Document ingestion and embedding
- `02_retrieval_pipeline.py` - Query retrieval
- `03_answer_generation.py` - Answer generation using LLM
- `04_history_aware_generation.py` - Chat interface with conversation history
- `05_semantic_chunking.py` - Semantic chunking example
- `06_agentic_chunking.py` - AI-powered chunking example
- `07_multi_modal_rag.ipynb` - Multi-modal RAG with text, images, and tables
- `08_retrieval_methods.py` - Different retrieval strategies comparison
- `09_multi_query_retrieval.py` - Multi-query retrieval with LLM-generated variations
- `10_reciprocal_rank_fusion.py` - RRF for combining multiple retrieval results
- `11_hybrid_search.ipynb` - Hybrid search combining semantic and keyword retrieval
- `12_reranker.ipynb` - Cohere reranker to improve retrieval relevance
- `db/chroma_db/` - Vector database (ignored by git)

## Dependencies

- LangChain
- ChromaDB
- Google Generative AI (embeddings)
- python-dotenv
