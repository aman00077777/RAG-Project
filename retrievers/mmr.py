from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma-db-local",
    embedding_function=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

docs = retriever.invoke("large language models")

for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)