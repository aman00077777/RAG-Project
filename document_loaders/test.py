from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    "document_loaders/notes.txt",
    encoding="utf-8"
)

docs = loader.load()

print(f"Documents loaded: {len(docs)}")
print(docs[0].page_content[:500])
