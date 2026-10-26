from langchain_community.document_loaders import TextLoader

data = TextLoader("Utsav_Kashyap_AIML_Cover_Letter.txt")

docs = data.load()
print(len(docs))