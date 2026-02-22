from fastapi import FastAPI, Body, File, UploadFile, Depends
from contextlib import asynccontextmanager

from typing import List, Annotated, Any

from ocr.pytesseract_engine import OCREngine
from persistence.db import DatabaseManager, get_db_manager
from persistence.repository import OCRRepository
from services.base import BaseService
from services.ingestion import IngestionService
from services.parsing import ParsingService
from llm.config import LLMConfig
from llm.registry import get_llm_provider
from llm.base import LLMProvider

# Global instances
db_manager: DatabaseManager | None = None
ocr_engine: OCREngine | None = None
ingestion_service: IngestionService | None = None
parsing_service: ParsingService | None = None
llm_provider: LLMProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database, OCR engine, and LLM provider lifecycle."""
    print("Application starting up...")
    global db_manager, ocr_engine, ingestion_service, parsing_service, llm_provider

    # Initialize database
    db_manager = get_db_manager()
    await db_manager.connect()

    # Initialize OCR engine
    ocr_engine = OCREngine(language="ara")

    # Initialize LLM provider from configuration
    llm_config = LLMConfig()
    print(f"Initializing LLM provider: {llm_config.provider} (model: {llm_config.model})")
    llm_provider = get_llm_provider(llm_config)

    # Initialize services
    ingestion_service = IngestionService(ocr_engine=ocr_engine)
    parsing_service = ParsingService(ocr_engine=ocr_engine, llm_provider=llm_provider)

    # Start services
    await ingestion_service.start()
    await parsing_service.start()

    yield

    # Shutdown services
    await parsing_service.stop()
    await ingestion_service.stop()
    await db_manager.disconnect()
    print("Application shutting down...")


app = FastAPI(lifespan=lifespan)


# Dependency Section:
def get_ocr_repository() -> OCRRepository:
    """Dependency function to get an instance of OCRRepository."""
    if db_manager is None:
        raise RuntimeError("Database manager is not initialized")
    return OCRRepository(db_manager)


OCRRepositoryDep = Annotated[OCRRepository, Depends(get_ocr_repository)]


def get_ingestion_service() -> IngestionService:
    """Dependency function to get an instance of IngestionService."""
    if ingestion_service is None:
        raise RuntimeError("Ingestion service is not initialized")
    return ingestion_service


IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]


def get_parsing_service() -> ParsingService:
    """Dependency function to get an instance of ParsingService."""
    if parsing_service is None:
        raise RuntimeError("Parsing service is not initialized")
    return parsing_service


ParsingServiceDep = Annotated[ParsingService, Depends(get_parsing_service)]


def get_llm_provider_dep() -> LLMProvider:
    """Dependency function to get an instance of LLMProvider."""
    if llm_provider is None:
        raise RuntimeError("LLM provider is not initialized")
    return llm_provider


LLMProviderDep = Annotated[LLMProvider, Depends(get_llm_provider_dep)]


# Endpoints Section:
@app.get("/users/{type}/{id}")
async def get_user(type: str, id: int):
    return {"type": type, "id": id}


@app.post("/users")
async def create_user(name: str = Body(...), age: int = Body(...)):
    return {"name": name, "age": age}


@app.post("/upload-file")
async def upload_file(
        repository: OCRRepositoryDep,
        ingestion_svc: IngestionServiceDep,
        file: UploadFile = File(...)
):
    results = await ingestion_svc.ingest(file=file)
    return {"results": results}


@app.post("/upload-files")
async def upload_files(
        repository: OCRRepositoryDep,
        ingestion_svc: IngestionServiceDep,
        files: List[UploadFile] = File(...)
):
    """
    Upload and process multiple files using the IngestionService.

    Each file is processed individually through the ingestion service,
    which handles both PDF and image files appropriately.
    """
    results = []
    for file in files:
        file_results = await ingestion_svc.ingest(file=file)
        results.append({
            "filename": file.filename,
            "content": file_results
        })

    return {"results": results}


@app.post("/parse-with-llm")
async def parse_with_llm(
        ingestion_svc: IngestionServiceDep,
        parsing_svc: ParsingServiceDep,
        file: UploadFile = File(...),
        prompt: str = Body(default=None),
):
    """
    Parse a document and enhance the extracted text using LLM.

    Requires LLM provider to be configured.
    """
    if not parsing_svc.has_llm:
        return {"error": "LLM provider not configured", "healthy": False}

    ingestion_svc.set_ocr_language("eng")  # For now... we can make this dynamic later
    content = await ingestion_service.ingest(file=file, force_ocr=False)
    content = "\n".join(content)
    print(f"Extracted content from '{file.filename}':\n{content}\n---")
    result = await parsing_svc.parse_with_llm(document_content=content, prompt=prompt)

    return {
        "filename": file.filename,
        "content": result,
        "llm_model": parsing_svc.llm_provider.name,
    }


@app.post("/enhance-text")
async def enhance_text(
        parsing_svc: ParsingServiceDep,
        text: str = Body(..., embed=True),
        instruction: str = Body(..., embed=True),
):
    """
    Enhance existing text using LLM.

    Args:
        text: The text to enhance.
        instruction: Instruction for enhancement (e.g., "Summarize", "Translate to Arabic").
    """
    if not parsing_svc.has_llm:
        return {"error": "LLM provider not configured", "healthy": False}

    result = await parsing_svc.enhance_with_llm(text=text, instruction=instruction)

    return {
        "original_length": len(text),
        "enhanced_text": result,
        "llm_model": parsing_svc.llm_provider.name,
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for all services."""
    health_status = {
        "database": {"healthy": False},
        "ingestion_service": {"healthy": False},
        "parsing_service": {"healthy": False},
        "llm_provider": {"healthy": False},
    }

    # Database health
    if db_manager:
        health_status["database"] = await db_manager.health_check()

    # Ingestion service health
    if ingestion_service:
        health_status["ingestion_service"] = await ingestion_service.health_check()

    # Parsing service health (includes LLM health)
    if parsing_service:
        health_status["parsing_service"] = await parsing_service.health_check()

    # LLM provider health
    if llm_provider:
        health_status["llm_provider"] = await llm_provider.health_check()

    # Overall health
    all_healthy = all(
        component.get("healthy", False)
        for component in health_status.values()
        if isinstance(component, dict)
    )

    return {
        "healthy": all_healthy,
        "status": "healthy" if all_healthy else "degraded",
        "components": health_status,
    }


@app.post("/mock-upload")
async def mock_upload(files: List[UploadFile] = File(...)):
    filenames = [file.filename for file in files]
    print(f"!!! MOCK SUCCESS: Received {len(filenames)} files: {filenames}")
    return {
        "status": "mock_success",
        "received_count": len(filenames),
        "file_names": filenames
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
