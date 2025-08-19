from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document

# class DocumentRetriever:
#     def __init__(self, doc_splits: List[Document]):
#         self.embeddings = OpenAIEmbeddings()
#         self.vector_store, self.retriever = self.create_vector_store(doc_splits)
#         self.retriever_tool = self.create_retriever()

#     def create_vector_store(self, doc_splits: List[Document]):
#         vector_store = InMemoryVectorStore.from_documents(
#             doc_splits,
#             embedding=self.embeddings
#         )
#         retriever = vector_store.as_retriever()

#         return vector_store, retriever
        
#     def create_retriever(self):
#         retriever_tool = create_retriever_tool(
#         self.retriever,
#         "retrieve_blog_posts",
#         "Search and return information about Lilian Weng blog posts.",
#         )

#         return retriever_tool

#     def invoke_retriever_tool(self, query: str):
#         return self.retriever_tool.invoke(query)


# =======================================================

from langchain_core.documents import Document

class DocumentRetriever:
    def __init__(self, doc_splits: List[Document]):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store, self.retriever = self.create_vector_store(doc_splits)
        self.retriever_tool = self.create_retriever()

    def create_vector_store(self, doc_splits: List[Document]):
        vector_store = InMemoryVectorStore.from_documents(
            doc_splits,
            embedding=self.embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # top-3
        return vector_store, retriever
        
    def create_retriever(self):
        return create_retriever_tool(
            self.retriever,
            "retrieve_blog_posts",
            "Search and return information about Lilian Weng blog posts.",
        )

    def invoke_retriever_tool(self, query: str):
        """This goes through the tool (might return strings)."""
        return self.retriever_tool.invoke(query)

    def retrieve_documents(self, query: str) -> List[Document]:
        """Direct retriever access – always returns Documents."""
        return self.retriever.get_relevant_documents(query)
