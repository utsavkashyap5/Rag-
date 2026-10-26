from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0
)
data = PyPDFLoader("document_loader/Deep_Learning_Module1_5Marks_Notes.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    separator=" ",
    chunk_size = 10,
    chunk_overlap=1
)

chunks  = splitter.split_documents(docs)

template = ChatPromptTemplate([
    ("system","you are an ai assistant that summarizes the text"),
    ("human", "{data}")
])

prompt = template.format_messages(data=docs)


response = llm.invoke(prompt)
print(response.content)