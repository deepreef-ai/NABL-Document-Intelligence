from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.llm.factory import get_llm_chain
from app.models import Application, ChatMessage, Document

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str


def _get_application(db: Session, application_id: str) -> Application:
    application = db.get(Application, application_id)
    if application is None:
        raise HTTPException(404, f"application {application_id!r} not found")
    return application


def _context_summary(documents: list[Document]) -> str:
    lines = []
    for d in documents:
        field_bits = ", ".join(f"{f.field_path}={f.value!r} (conf {f.confidence:.2f})" for f in d.fields)
        lines.append(f"- {d.filename} [{d.doc_type or 'unclassified'}]: {field_bits or 'no fields extracted'}")
    return "\n".join(lines) or "No documents uploaded yet."


@router.get("/applications/{application_id}/chat")
def get_chat_history(application_id: str, db: Session = Depends(get_db)):
    _get_application(db, application_id)
    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.application_id == application_id)
        .order_by(ChatMessage.created_at)
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in history]


@router.post("/applications/{application_id}/chat")
def send_chat_message(application_id: str, body: ChatRequest, db: Session = Depends(get_db)):
    application = _get_application(db, application_id)
    documents = db.query(Document).filter(Document.application_id == application_id).all()

    system = (
        "You are the NABL Document Intelligence assistant, available in a "
        "side-drawer while the user reviews their auto-filled application. "
        "Answer questions about what was extracted and from where, and "
        "suggest using the re-extract button on a document if the user "
        "reports a value is wrong. Be concise."
    )
    prompt = (
        f"Application form_type={application.form_type}\n\n"
        f"Documents so far:\n{_context_summary(documents)}\n\n"
        f"User question: {body.message}"
    )
    reply = get_llm_chain().generate_text(system=system, user_text=prompt)

    db.add(ChatMessage(application_id=application_id, role="user", content=body.message))
    db.add(ChatMessage(application_id=application_id, role="assistant", content=reply))
    db.commit()
    return {"reply": reply}
