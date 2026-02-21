"""Galaxy viewer API — graph data and paper detail endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..deps import CurrentUser, get_current_user
from ..schemas import GalaxyData, PaperDetail
from ..services.galaxy_service import build_galaxy_data, get_paper_detail

router = APIRouter()


@router.get("/data", response_model=GalaxyData)
def get_galaxy_data(
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    data = build_galaxy_data(db, user_id=current_user.id)
    return GalaxyData(**data)


@router.get("/paper/{paper_id}", response_model=PaperDetail)
def get_galaxy_paper(
    paper_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    detail = get_paper_detail(db, paper_id, user_id=current_user.id)
    if not detail:
        raise HTTPException(404, "Paper not found")
    return PaperDetail(**detail)
