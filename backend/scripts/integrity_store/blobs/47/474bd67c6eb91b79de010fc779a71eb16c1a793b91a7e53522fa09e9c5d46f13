from pydantic import BaseModel


class PrerequisiteQuestion(BaseModel):
    id: str
    text: str
    help_text: str | None = None


class PrerequisiteAnswer(BaseModel):
    question_id: str
    satisfied: bool | None = None  # None = not yet answered / unclear
    detail: str | None = None


class EligibilityState(BaseModel):
    form_type: str
    answers: dict[str, PrerequisiteAnswer]
    all_satisfied: bool
    next_question: PrerequisiteQuestion | None = None
