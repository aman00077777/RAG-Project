import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

loader = PyPDFLoader(
    "document_loaders/GRU.pdf"
)

docs = loader.load()

if not docs or not docs[0].page_content.strip():
    raise ValueError("The document is empty or could not be loaded.")

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant that summarizes text accurately. "
            "Preserve important information and avoid adding facts "
            "that are not present in the original text."
        ),
        (
            "human",
            "Summarize the following text in simple, clear language. "
            "Use headings and bullet points where appropriate. "
            "Include the key concepts and important details.\n\n"
            "{data}"
        )
    ]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_tokens=1024,
    temperature=0.2,
)

chain = template | model

response = chain.invoke(
    {"data": docs       }
)

print(response.content)