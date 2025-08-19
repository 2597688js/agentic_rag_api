from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

class DocumentSplitter:
    def __init__(self):
        try:
            from src.config import ConfigManager
            self.config = ConfigManager()._config
            chunk_size = self.config.get('document', {}).get('chunk_size', 1000)
            chunk_overlap = self.config.get('document', {}).get('chunk_overlap', 200)
        except Exception as e:
            print(f"Failed to load config in document_splitter.py: {e}, using defaults")
            chunk_size = 1000
            chunk_overlap = 200
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, docs_list: List[Document]):
        return self.text_splitter.split_documents(docs_list)
