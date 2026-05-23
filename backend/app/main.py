from fastapi import FastAPI

from app.api.routes.analyze import router as analyze_router

app = FastAPI(
    title="LegalExplain-CH API",
    description="Source-grounded legal issue mapping API for Swiss employment law.",
    version="0.1.0",
)

app.include_router(analyze_router)


@app.get("/")
def read_root():
    return {
        "message": "LegalExplain-CH backend is running.",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
    }