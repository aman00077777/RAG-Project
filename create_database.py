from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma-db-local",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.7
    }
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
Use the following context to answer the question.
If the answer is not present in the context, say "I don't know based on the provided document."
Do not invent facts."""
        ),
        (
            "human",
            "Context: {context}\n\nQuestion: {question}"
        )
    ]
)

print("RAG System Created")
print("Press 0 to exit")

while True:
    query = input("Your Question: ").strip()

    if query == "0":
        break

    if not query:
        continue

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt_input = prompt.format_messages(
        context=context,
        question=query
    )

    response = llm.invoke(prompt_input)

    print("\nResponse:", response.content)