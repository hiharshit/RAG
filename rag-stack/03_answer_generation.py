from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from dotenv import load_dotenv


load_dotenv()

persistent_directory = "db/chroma_db"
# load embedding model and vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# retreive relevant docs
user_query = input("Please enter your query")

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.7},
)

relevant_docs = retriever.invoke(user_query)

# combine user input and the context from the relevant docs
combined_prompt = f"""
Based on the following documents, answer thsi question: {user_query}
Documents:
        {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""


# create a ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_prompt),
]


# invoke the model using the combined input
result = model.invoke(messages)
print("\n---Generated Response---")
# print out the result
# print("Full result:")
# print(result)

print("Content only:")
print(result.content)
