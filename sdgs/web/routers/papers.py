"""Papers API: list and search papers used in dataset generation."""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import Paper
from ..deps import CurrentUser, get_current_user
from ..schemas import PaperListResponse, PaperResponse

router = APIRouter()


@router.get("", response_model=PaperListResponse)
def list_papers(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    search: str | None = None,
    dataset_id: int | None = None,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Paper).filter(Paper.user_id == current_user.id)

    if dataset_id is not None:
        query = query.filter(Paper.dataset_id == dataset_id)

    if search:
        pattern = f"%{search}%"
        query = query.filter(
            Paper.title.ilike(pattern)
            | Paper.paper_id.ilike(pattern)
        )

    total = query.count()
    papers = (
        query.order_by(Paper.id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return PaperListResponse(
        papers=[PaperResponse.model_validate(p) for p in papers],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{paper_id}/pdf")
def download_paper_pdf(
    paper_id: int,
    current_user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter(
        Paper.id == paper_id,
        Paper.user_id == current_user.id,
    ).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if not paper.pdf_path:
        raise HTTPException(status_code=404, detail="No PDF available for this paper")

    pdf_file = Path(paper.pdf_path)
    if not pdf_file.is_file():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    safe_title = paper.title[:80].replace('"', "'")
    return FileResponse(
        path=str(pdf_file),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_title}.pdf"'},
    )
