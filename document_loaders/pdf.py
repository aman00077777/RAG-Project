from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document_loaders/GRU.pdf").load()

print(f"Number of pages: {len(data)}")

print(data[0])

print(data[1])

print(data[2])

