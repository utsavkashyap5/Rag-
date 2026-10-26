from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

data = TextLoader("./text.txt")

docs = data.load()

spliiter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap =20,

)

chunks = spliiter.split_documents(docs)



embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
persist_dir = "./chroma_db"

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

print(f"Success! Successfully uploaded {len(chunks)} chunks into Chroma DB.")