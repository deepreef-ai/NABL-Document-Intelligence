import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Application(Base):
    """One NABL accreditation application: the wizard-created record that
    documents get uploaded against and the form gets filled for."""

    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    form_type: Mapped[str] = mapped_column(String)  # NABL_151 / NABL_152 / NABL_153
    # "eligibility" -> "unlocked" -> "reviewing" -> "complete"
    status: Mapped[str] = mapped_column(String, default="eligibility")
    prerequisite_answers: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    documents: Mapped[list["Document"]] = relationship(back_populates="application")
    chat_messages: Mapped[list["ChatMessage"]] = relationship(back_populates="application")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"))
    filename: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String)
    storage_path: Mapped[str] = mapped_column(String)
    # populated by the classifier once the pipeline runs
    doc_type: Mapped[str | None] = mapped_column(String, nullable=True)
    extraction_source: Mapped[str | None] = mapped_column(String, nullable=True)
    # "uploaded" -> "processing" -> "extracted" -> "failed"
    status: Mapped[str] = mapped_column(String, default="uploaded")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    application: Mapped[Application] = relationship(back_populates="documents")
    fields: Mapped[list["ExtractedField"]] = relationship(back_populates="document")


class ExtractedField(Base):
    """One extracted value, e.g. field_path='equipment[0].serial_number'."""

    __tablename__ = "extracted_fields"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    field_path: Mapped[str] = mapped_column(String)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source_bbox: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source_page: Mapped[int | None] = mapped_column(nullable=True)
    # "llm" | "rule_based" | "verification" — see documents/grounding.py's
    # FieldResult and documents/extraction_report.py.
    source: Mapped[str] = mapped_column(String, default="llm")
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)

    document: Mapped[Document] = relationship(back_populates="fields")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"))
    role: Mapped[str] = mapped_column(String)  # "user" / "assistant" / "system"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    application: Mapped[Application] = relationship(back_populates="chat_messages")
