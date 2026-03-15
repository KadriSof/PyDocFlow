import logging
from fastapi import FastAPI, Body, File, UploadFile, Depends, Query, HTTPException
from contextlib import asynccontextmanager
from typing import List, Annotated, Any

from ocr.pytesseract_engine import OCREngine
from persistence.base import BaseDocumentRepository
from persistence.factory import DatabaseFactory
from services.ingestion import IngestionService
from services.parsing import ParsingService
from services.persistence import PersistenceService
from llm.config import LLMConfig
from llm.registry import get_llm_provider
from llm.base import LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (managed via factory/lifespan)
ocr_engine: OCREngine | None = None
ingestion_service: IngestionService | None = None
parsing_service: ParsingService | None = None
llm_provider: LLMProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage database, OCR engine, and LLM provider lifecycle."""
    logger.info("Application starting up...")
    global ocr_engine, ingestion_service, parsing_service, llm_provider

    # Initialize databases
    try:
        await DatabaseFactory.get_client("postgresql")
        logger.info("PostgreSQL client initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize PostgreSQL client: {e}")

    try:
        await DatabaseFactory.get_client("mongodb")
        logger.info("MongoDB client initialized.")
    except Exception as e:
        logger.warning(f"Failed to initialize MongoDB client: {e}")

    # Initialize OCR engine
    ocr_engine = OCREngine(language="ara")

    # Initialize LLM provider from configuration
    llm_config = LLMConfig()
    logger.info(f"Initializing LLM provider: {llm_config.provider} (model: {llm_config.model})")
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
    await DatabaseFactory.disconnect_all()
    logger.info("Application shutting down...")


app = FastAPI(
    title="PyDocFlow API",
    description="Document processing service with OCR and LLM-powered text extraction",
    version="0.1.0",
    lifespan=lifespan
)


# Dependency Section:
def get_document_repository(
    db: Annotated[str, Query(description="Database type: 'postgresql' or 'mongodb'")] = "postgresql"
) -> BaseDocumentRepository:
    """
    Dependency to get a DocumentRepository.
    Supports both PostgreSQL and MongoDB based on the 'db' query parameter.
    """
    try:
        return DatabaseFactory.get_repository(db)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


DocumentRepoDep = Annotated[BaseDocumentRepository, Depends(get_document_repository)]


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


def get_persistence_service(
    repository: DocumentRepoDep,
) -> PersistenceService:
    """Dependency function to get an instance of PersistenceService."""
    svc = PersistenceService(repository=repository)
    svc._is_running = True  # Dependencies are already running
    return svc


PersistenceServiceDep = Annotated[PersistenceService, Depends(get_persistence_service)]


# Endpoints Section:

@app.post("/upload-file", tags=["Ingestion"])
async def upload_file(
    ingestion_svc: IngestionServiceDep,
    persistence_svc: PersistenceServiceDep,
    file: Annotated[UploadFile, File(...)]
):
    """
    Ingest a file and save it to the repository.
    """
    # 1. Ingest (OCR/Text Extraction)
    texts = await ingestion_svc.ingest(file=file)
    
    # 2. Persist results
    document = await persistence_svc.persist_document(file=file, texts=texts)
    
    return {
        "filename": file.filename,
        "document_id": document.file_id,
        "page_count": len(document.pages),
    }


@app.post("/upload-files", tags=["Ingestion"])
async def upload_files(
    ingestion_svc: IngestionServiceDep,
    persistence_svc: PersistenceServiceDep,
    files: Annotated[List[UploadFile], File(...)]
):
    """
    Upload and process multiple files.
    """
    results = []
    for file in files:
        texts = await ingestion_svc.ingest(file=file)
        document = await persistence_svc.persist_document(file=file, texts=texts)
        results.append({
            "filename": file.filename,
            "document_id": document.file_id,
            "page_count": len(document.pages)
        })

    return {"results": results}


@app.post(
    "/parse-with-llm", 
    tags=["Parsing"],
    responses={503: {"description": "LLM provider not configured"}}
)
async def parse_with_llm(
    ingestion_svc: IngestionServiceDep,
    parsing_svc: ParsingServiceDep,
    persistence_svc: PersistenceServiceDep,
    file: Annotated[UploadFile, File(...)],
    prompt: Annotated[str | None, Body()] = None,
):
    """
    Parse a document and enhance the extracted text using LLM.
    """
    if not parsing_svc.has_llm:
        raise HTTPException(status_code=503, detail="LLM provider not configured")

    # 1. Ingest
    texts = await ingestion_svc.ingest(file=file)
    content = "\n".join(texts)
    
    # 2. Enhance with LLM
    enhanced_content = await parsing_svc.parse_with_llm(document_content=content, prompt=prompt)

    # 3. Persist (original + enhanced in metadata)
    metadata = {
        "llm_enhanced": True,
        "llm_model": parsing_svc.llm_provider.name if parsing_svc.llm_provider else "unknown",
        "enhanced_content": enhanced_content
    }
    document = await persistence_svc.persist_document(file=file, texts=texts, metadata=metadata)

    return {
        "filename": file.filename,
        "document_id": document.file_id,
        "content": enhanced_content,
        "llm_model": parsing_svc.llm_provider.name if parsing_svc.llm_provider else "unknown",
    }


@app.post("/enhance-text", tags=["Parsing"])
async def enhance_text(
    parsing_svc: ParsingServiceDep,
    text: Annotated[str, Body(embed=True)],
    instruction: Annotated[str, Body(embed=True)],
):
    """Enhance existing text using LLM."""
    if not parsing_svc.has_llm:
        raise HTTPException(status_code=503, detail="LLM provider not configured")

    result = await parsing_svc.enhance_with_llm(text=text, instruction=instruction)

    return {
        "original_length": len(text),
        "enhanced_text": result,
        "llm_model": parsing_svc.llm_provider.name if parsing_svc.llm_provider else "unknown",
    }


@app.get("/documents", tags=["Persistence"])
async def list_documents(
    repository: DocumentRepoDep, 
    skip: Annotated[int, Query(ge=0)] = 0, 
    limit: Annotated[int, Query(le=100)] = 10
):
    """List all processed documents from the specified database."""
    return await repository.list_all(skip=skip, limit=limit)


@app.get("/health", tags=["System"])
async def health_check(
    persistence_svc: PersistenceServiceDep,
) -> dict[str, Any]:
    """Health check endpoint for all services and databases."""
    health_status = {
        "databases": {},
        "ingestion_service": {"healthy": False},
        "parsing_service": {"healthy": False},
        "persistence_service": {"healthy": False},
        "llm_provider": {"healthy": False},
    }

    # Database health
    active_clients = DatabaseFactory.get_active_clients()
    for db_type, client in active_clients.items():
        health_status["databases"][db_type] = await client.health_check()

    # Ingestion service health
    if ingestion_service:
        health_status["ingestion_service"] = await ingestion_service.health_check()

    # Parsing service health
    if parsing_service:
        health_status["parsing_service"] = await parsing_service.health_check()

    # Persistence service health
    if persistence_svc:
        health_status["persistence_service"] = await persistence_svc.health_check()

    # LLM provider health
    if llm_provider:
        health_status["llm_provider"] = await llm_provider.health_check()

    # Overall health
    db_healthy = len(health_status["databases"]) > 0 and all(
        db.get("healthy", False) for db in health_status["databases"].values()
    )
    others_healthy = all(
        health_status[k].get("healthy", False) 
        for k in ["ingestion_service", "parsing_service", "persistence_service", "llm_provider"]
    )
    all_healthy = db_healthy and others_healthy

    return {
        "healthy": all_healthy,
        "status": "healthy" if all_healthy else "degraded",
        "components": health_status,
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
