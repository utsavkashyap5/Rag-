from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("Deep_Learning_Module1_5Marks_Notes.pdf")

docs = data.load()

print(len(docs))