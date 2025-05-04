from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
import os 
from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                
        
if __name__ == "__main__":
    main()