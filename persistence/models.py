from odmantic import Model, Field
from datetime import datetime


class OCRResult(Model):
    file_name: str = Field(primary_field=True)
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)
