from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import ConfigManager
from typing import List
from langchain_core.documents import Document

class DocumentSplitter:
    def __init__(self):
        self.config = ConfigManager()._config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.config['document_processing']['chunk_size'], 
            chunk_overlap = self.config['document_processing']['chunk_overlap']
            )

    def split_documents(self, docs_list: List[Document]):
        return self.text_splitter.split_documents(docs_list)
