import urllib.parse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongo_user: str = Field(default=..., validation_alias="MONGO_USER")
    mongo_password: str = Field(default=..., validation_alias="MONGO_PASSWORD")
    mongo_host: str = Field(default="localhost", validation_alias="MONGO_HOST")
    mongo_port: str = Field(default="27017", validation_alias="MONGO_PORT")
    mongo_db: str = Field(default="ocr_db", validation_alias="MONGO_DB")

    @property
    def mongo_uri(self) -> str:
        """Construct the MongoDB URI from the settings."""
        user = urllib.parse.quote_plus(self.mongo_user)
        password = urllib.parse.quote_plus(self.mongo_password)
        return f"mongodb://{user}:{password}@{self.mongo_host}:{self.mongo_port}?authSource=admin"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


if __name__ == "__main__":
    settings = Settings()
    print("MongoDB URI:", settings.mongo_uri)
