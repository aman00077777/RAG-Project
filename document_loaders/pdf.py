from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("document_loaders/GRU.pdf").load()

print(f"Number of pages: {len(data)}")

splitter = TokenTextSplitter(
    chunk_size=300,
    chunk_overlap=70,
    encoding_name="gpt2"
)

chunks = splitter.split_documents(data)

print(f"Number of chunks: {len(chunks)}")
print(chunks[0])