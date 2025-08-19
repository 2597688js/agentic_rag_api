from pydantic import BaseModel, Field
from typing import List, Optional

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class RAGRequest(BaseModel):
    query: str
    file_paths_urls: List[str]

class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    nodes_executed: List[str]