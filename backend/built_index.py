import os, pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

with open("data/rental_qa.txt", "r", encoding="utf-8") as f:
    content = f.read()

docs = [Document(page_content=s.strip()) for s in content.split("\n\n") if s.strip()]
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
split_docs = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vectorstore = FAISS.from_documents(split_docs, embedding)

os.makedirs("vectorstore", exist_ok=True)
with open("vectorstore/faiss_index.pkl", "wb") as f:
    pickle.dump((vectorstore.index, [d.page_content for d in split_docs]), f)
