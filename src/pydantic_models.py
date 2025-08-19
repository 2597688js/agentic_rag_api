from pydantic import BaseModel, Field
from typing import List, Optional

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class FileContent(BaseModel):
    name: str
    content: str  # base64 encoded content
    type: str = "file"

class RAGRequest(BaseModel):
    query: str
    file_paths_urls: List[str | FileContent]  # Can be URLs or file content objects

class RAGResponse(BaseModel):
    response: str
    top_3_retrieved_docs: List[str]
    metadata: List[dict]

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    message: str