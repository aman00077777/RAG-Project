from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

data = TextLoader(
    "document_loaders/notes.txt",
    encoding="utf-8"
).load()

split = splitter.split_documents(data)

print("Total chunks:", len(split))
print("\nFirst chunk:\n")
print(split[0].page_content)
print("\nMetadata:", split[0].metadata)