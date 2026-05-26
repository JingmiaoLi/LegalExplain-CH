from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.analyze import router as analyze_router
from backend.app.api.routes.ask import router as ask_router

app = FastAPI(
    title="LegalExplain-CH API",
    description="Source-grounded legal issue mapping API for Swiss employment law.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)
app.include_router(ask_router)


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