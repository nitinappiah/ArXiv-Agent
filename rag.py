from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings

def create_vector_store(entries, model_info):
    documents = []
    for entry in entries:
        link = entry['id']
        text = entry['summary']
        document = Document(
            page_content=text, 
            metadata={
                "source": str(entry['id']), 
                "published": str(entry['published']), 
                "authors": str(entry['author']), 
                "category": str(entry['category'])
            }
        )
        documents.append(document)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    documents = text_splitter.split_documents(documents)
    if model_info['type'] == 'openai':
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=model_info['api_key'])
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    db = Chroma.from_documents(documents, embeddings)

    return db

def retrieve_documents(db, query, model_info):
    if model_info['type'] == 'openai':
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=model_info['api_key'])
    else:
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k":100, "score_threshold":0.4}
    )
    relevant_docs = retriever.invoke(query)
    return relevant_docs