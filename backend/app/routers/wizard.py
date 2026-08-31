from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Application, ChatMessage
from app.schemas.forms import NablFormType
from app.wizard.engine import WizardEngine, compute_state

router = APIRouter(tags=["wizard"])


class CreateApplicationRequest(BaseModel):
    form_type: NablFormType


class AnswerRequest(BaseModel):
    message: str


def _get_application(db: Session, application_id: str) -> Application:
    application = db.get(Application, application_id)
    if application is None:
        raise HTTPException(404, f"application {application_id!r} not found")
    return application


@router.post("/applications")
def create_application(body: CreateApplicationRequest, db: Session = Depends(get_db)):
    application = Application(form_type=body.form_type.value)
    db.add(application)
    db.commit()
    db.refresh(application)

    state, message = WizardEngine(db).start(application)
    return {"application": _serialize(application), "state": state, "message": message}


@router.get("/applications/{application_id}/wizard")
def get_wizard_state(application_id: str, db: Session = Depends(get_db)):
    application = _get_application(db, application_id)
    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.application_id == application_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    return {
        "application": _serialize(application),
        "state": compute_state(application),
        "history": [{"role": m.role, "content": m.content} for m in history],
    }


@router.post("/applications/{application_id}/wizard/answer")
def answer_wizard_question(application_id: str, body: AnswerRequest, db: Session = Depends(get_db)):
    application = _get_application(db, application_id)
    state, reply = WizardEngine(db).submit_answer(application, body.message)
    return {"application": _serialize(application), "state": state, "reply": reply}


def _serialize(application: Application) -> dict:
    return {
        "id": application.id,
        "form_type": application.form_type,
        "status": application.status,
        "created_at": application.created_at,
    }
