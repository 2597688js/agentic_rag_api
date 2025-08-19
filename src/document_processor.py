from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from src.config import ConfigManager

class DocumentProcessor:
    def __init__(self):
        self.config = ConfigManager()._config

    def load_documents(self, sources: List[str]):
        docs_list = []  # collect all documents in one flat list
        errors = []  # collect any loading errors
        
        for source in sources:
            try:
                if source.startswith("http://") or source.startswith("https://"):
                    try:
                        loader = WebBaseLoader(source)
                        docs = loader.load()
                        if docs:
                            docs_list.extend(docs)
                            print(f"✅ Successfully loaded URL: {source}")
                        else:
                            errors.append(f"No content extracted from URL: {source}")
                    except Exception as url_error:
                        errors.append(f"Failed to load URL {source}: {str(url_error)}")
                        continue
                        
                elif source.endswith(".pdf"):
                    # Check if file exists before trying to load
                    import os
                    if not os.path.exists(source):
                        raise FileNotFoundError(f"File not found: {source}")
                    loader = PyPDFLoader(source)
                    docs = loader.load()
                    if docs:
                        docs_list.extend(docs)
                        print(f"✅ Successfully loaded PDF: {source}")
                    else:
                        errors.append(f"No content extracted from PDF: {source}")
                        
                elif source.endswith(".docx") or source.endswith(".doc"):
                    import os
                    if not os.path.exists(source):
                        raise FileNotFoundError(f"File not found: {source}")
                    loader = Docx2txtLoader(source)
                    docs = loader.load()
                    if docs:
                        docs_list.extend(docs)
                        print(f"✅ Successfully loaded DOCX: {source}")
                    else:
                        errors.append(f"No content extracted from DOCX: {source}")
                        
                elif source.endswith(".txt"):
                    import os
                    if not os.path.exists(source):
                        raise FileNotFoundError(f"File not found: {source}")
                    loader = TextLoader(source)
                    docs = loader.load()
                    if docs:
                        docs_list.extend(docs)
                        print(f"✅ Successfully loaded TXT: {source}")
                    else:
                        errors.append(f"No content extracted from TXT: {source}")
                        
                else:
                    raise ValueError(f"Unsupported file or URL format: {source}")
                    
            except Exception as e:
                error_msg = f"Failed to load {source}: {str(e)}"
                errors.append(error_msg)
                continue
        
        # If we have errors, log them but continue with what we could load
        if errors:
            print(f"Warning: {len(errors)} source(s) failed to load, but continuing with {len(docs_list)} successful documents")
            for error in errors:
                print(f"  - {error}")
        
        return docs_list

