from odmantic import Model, EmbeddedModel, Field, Index
from datetime import datetime
from typing import ClassVar, List


class Page(EmbeddedModel):
    """Embedded model representing a single page within a document."""

    page_number: int
    content: str
    metadata: dict = Field(default_factory=dict)


class Document(Model):
    """Model representing a document with one or multiple pages."""

    file_id: str = Field(primary_field=True)
    file_name: str
    pages: List[Page]
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)

    model_config: ClassVar[dict] = {  # type: ignore[assignment]
        "indexes": lambda: [
            Index(Document.file_name, unique=True),
            Index(Document.created_at),
        ]
    }
