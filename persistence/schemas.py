from typing import List, Dict, Any
from pydantic import BaseModel, Field


class PageCreate(BaseModel):
    """Schema for creating a document page."""
    page_number: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(BaseModel):
    """Schema for creating a new document with its pages."""
    file_id: str
    file_name: str
    pages: List[PageCreate]
    metadata: Dict[str, Any] = Field(default_factory=dict)
