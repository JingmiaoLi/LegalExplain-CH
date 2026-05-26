from fastapi import APIRouter

from backend.app.schemas.legal import AnalyzeRequest, AnalyzeResponse
from backend.app.services.workflow import run_legal_workflow

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("", response_model=AnalyzeResponse)
def analyze_scenario(request: AnalyzeRequest) -> AnalyzeResponse:
    return run_legal_workflow(request.scenario)