from sqlalchemy.orm import Session

from app.llm.factory import get_llm_chain
from app.models import Application, ChatMessage
from app.schemas.eligibility import EligibilityState, PrerequisiteAnswer, PrerequisiteQuestion
from app.schemas.forms import NablFormType
from app.wizard.prerequisites import PREREQUISITES

GREETING = (
    "Let's check you're ready to apply for {form_type}. I'll ask a few mandatory "
    "eligibility questions first — once everything checks out, you can move on to "
    "uploading documents."
)

_SYSTEM = (
    "You are the NABL accreditation eligibility wizard. You ask mandatory "
    "prerequisite questions one at a time and must record a clear "
    "satisfied/not-satisfied verdict from the user's free-text reply, erring "
    "on the side of null (needs clarification) rather than guessing."
)


def _questions_for(form_type: str) -> list[PrerequisiteQuestion]:
    return PREREQUISITES[NablFormType(form_type)]


def compute_state(application: Application) -> EligibilityState:
    questions = _questions_for(application.form_type)
    answers = {qid: PrerequisiteAnswer(**data) for qid, data in (application.prerequisite_answers or {}).items()}
    for q in questions:
        answers.setdefault(q.id, PrerequisiteAnswer(question_id=q.id))
    next_question = next((q for q in questions if not answers[q.id].satisfied), None)
    return EligibilityState(
        form_type=application.form_type,
        answers=answers,
        all_satisfied=next_question is None,
        next_question=next_question,
    )


def _parse_answer(question: PrerequisiteQuestion, message: str) -> dict:
    prompt = (
        f"Question: {question.text}\nUser's reply: {message}\n\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"satisfied": true|false|null, "detail": "one-line paraphrase of what the user said", '
        '"reply": "a short, friendly chat reply"}\n'
        "satisfied: true if clearly met, false if clearly not met, null if unclear and needs a follow-up. "
        "reply: acknowledge the answer, and if not satisfied, explain what's missing and what to do next; "
        "if satisfied, lead into the next question or confirm the wizard is unlocked."
    )
    return get_llm_chain().generate_json(system=_SYSTEM, user_text=prompt)


class WizardEngine:
    def __init__(self, db: Session):
        self.db = db

    def start(self, application: Application) -> tuple[EligibilityState, str]:
        state = compute_state(application)
        message = GREETING.format(form_type=application.form_type)
        if state.next_question:
            message += f"\n\n{state.next_question.text}"
        self.db.add(ChatMessage(application_id=application.id, role="assistant", content=message))
        self.db.commit()
        return state, message

    def submit_answer(self, application: Application, message: str) -> tuple[EligibilityState, str]:
        state = compute_state(application)
        self.db.add(ChatMessage(application_id=application.id, role="user", content=message))

        question = state.next_question
        if question is None:
            reply = "All eligibility checks are already satisfied — you can move on to document upload."
            self.db.add(ChatMessage(application_id=application.id, role="assistant", content=reply))
            self.db.commit()
            return state, reply

        parsed = _parse_answer(question, message)

        answers = dict(application.prerequisite_answers or {})
        answers[question.id] = PrerequisiteAnswer(
            question_id=question.id, satisfied=parsed["satisfied"], detail=parsed.get("detail")
        ).model_dump(mode="json")
        application.prerequisite_answers = answers

        new_state = compute_state(application)
        if new_state.all_satisfied:
            application.status = "unlocked"

        reply = parsed["reply"]
        self.db.add(ChatMessage(application_id=application.id, role="assistant", content=reply))
        self.db.add(application)
        self.db.commit()
        self.db.refresh(application)
        return compute_state(application), reply
