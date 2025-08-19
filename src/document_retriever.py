from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, doc_splits: List[Document]):
        """Initialize the document retriever with document splits."""
        if not doc_splits:
            raise ValueError("No document splits provided")
        
        logger.info(f"Initializing retriever with {len(doc_splits)} document splits")
        
        # Initialize embeddings
        try:
            self.embeddings = OpenAIEmbeddings()
            logger.info("OpenAI embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise
        
        # Create vector store and retriever
        self.vector_store, self.retriever = self.create_vector_store(doc_splits)
        
        # Create retriever tool
        self.retriever_tool = self.create_retriever()
        
        logger.info("Document retriever initialized successfully")

    def create_vector_store(self, doc_splits: List[Document]):
        """Create vector store from document splits."""
        try:
            logger.info(f"Creating vector store with {len(doc_splits)} documents")
            
            # Create vector store
            vector_store = InMemoryVectorStore.from_documents(
                doc_splits,
                embedding=self.embeddings
            )
            logger.info("Vector store created successfully")
            
            # Create retriever with proper configuration
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,  # Retrieve top 5 documents
                }
            )
            logger.info("Retriever created with similarity search")
            
            return vector_store, retriever
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def create_retriever(self):
        """Create retriever tool using LangChain's create_retriever_tool."""
        try:
            retriever_tool = create_retriever_tool(
                self.retriever,
                "document_retriever",
                "Search and retrieve relevant information from uploaded documents and URLs.",
            )
            logger.info("Retriever tool created successfully")
            return retriever_tool
            
        except Exception as e:
            logger.error(f"Failed to create retriever tool: {e}")
            raise

    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """Direct retriever access - returns Documents."""
        try:
            logger.info(f"Retrieving documents for query: {query[:100]}...")
            
            # Use the retriever directly with proper parameters
            documents = self.retriever.get_relevant_documents(
                query,
                callbacks=None,
                tags=None,
                metadata=None,
                run_name=None
            )
            
            # Limit to k documents if more are returned
            if len(documents) > k:
                documents = documents[:k]
                logger.info(f"Limited results to top {k} documents")
            
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Log retrieved documents for debugging
            for i, doc in enumerate(documents):
                logger.debug(f"Document {i+1}: {doc.metadata.get('source', 'unknown')} - {doc.page_content[:100]}...")
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []

    def invoke_retriever_tool(self, query: str):
        """Invoke the retriever tool (returns tool output)."""
        try:
            logger.info(f"Invoking retriever tool for query: {query[:100]}...")
            result = self.retriever_tool.invoke(query)
            logger.info(f"Retriever tool returned: {type(result)}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to invoke retriever tool: {e}")
            return None

    def test_retriever(self, query: str = "test query"):
        """Test the retriever functionality."""
        try:
            logger.info("Testing retriever functionality...")
            
            # Test direct retrieval
            docs = self.retrieve_documents(query)
            logger.info(f"Direct retrieval test: {len(docs)} documents")
            
            # Test tool invocation
            tool_result = self.invoke_retriever_tool(query)
            logger.info(f"Tool invocation test: {type(tool_result)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Retriever test failed: {e}")
            return False
