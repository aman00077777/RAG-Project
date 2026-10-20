from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import MultiQueryRetriever

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma-db-local",
    embedding_function=embedding_model
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 3}
    ),
    llm=llm
)

docs = retriever.invoke("What are large language models?")

for i, doc in enumerate(docs):
    print(f"\nResult {i + 1}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)