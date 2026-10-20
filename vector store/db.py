from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is widely used in Artificial Intelligence.",
        metadata={"source": "AI_book"}
    ),
    Document(
        page_content="Pandas is used for data analysis in Python.",
        metadata={"source": "DataScience_book"}
    ),
    Document(
        page_content="Neural networks are used in deep learning.",
        metadata={"source": "DL_book"}
    )
]

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

print("Documents stored successfully!")

results = vector_store.similarity_search(
    "What is used in deep learning?",
    k=2
)

for result in results:
    print(result)

retriever = vector_store.as_retriever()

docs = retriever.invoke("What is used in deep learning?")

for d in docs:
    print(d.page_content)