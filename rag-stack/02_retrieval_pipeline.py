from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# load embdeddings and vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:spcae": "cosine"},
)

# search for the user query
user_query = input("Please enter your query: ")

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.7},
)

relevant_docs = retriever.invoke(user_query)

print(f"Your query: {user_query}")
# display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document: {i}:\n{doc.page_content}\n")
