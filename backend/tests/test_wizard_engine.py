from types import SimpleNamespace

import pytest

from app.schemas.forms import FORM_CATEGORY, NablFormType
from app.wizard.engine import compute_state
from app.wizard.prerequisites import PREREQUISITES


def _app(form_type="NABL_151", answers=None):
    return SimpleNamespace(form_type=form_type, prerequisite_answers=answers or {})


@pytest.mark.parametrize("form_type", list(NablFormType))
def test_every_form_type_has_prerequisite_questions(form_type):
    questions = PREREQUISITES[form_type]
    assert len(questions) >= 2
    assert len({q.id for q in questions}) == len(questions)  # no duplicate ids


def test_recognition_forms_get_different_questions_than_accreditation_forms():
    accreditation_ids = {q.id for q in PREREQUISITES[NablFormType.NABL_151]}
    for form_type, category in FORM_CATEGORY.items():
        if category.value == "recognition":
            recognition_ids = {q.id for q in PREREQUISITES[form_type]}
            assert recognition_ids != accreditation_ids


def test_first_question_is_standard_implemented_when_unanswered():
    state = compute_state(_app())
    assert state.all_satisfied is False
    assert state.next_question.id == "standard_implemented"


def test_advances_to_next_question_once_satisfied():
    answers = {"standard_implemented": {"question_id": "standard_implemented", "satisfied": True, "detail": "yes"}}
    state = compute_state(_app(answers=answers))
    assert state.next_question.id == "internal_audit_mrm"


def test_all_satisfied_when_every_prerequisite_is_true():
    ids = ["standard_implemented", "internal_audit_mrm", "pt_ilc", "quality_manager_staff"]
    answers = {qid: {"question_id": qid, "satisfied": True, "detail": "yes"} for qid in ids}
    state = compute_state(_app(answers=answers))
    assert state.all_satisfied is True
    assert state.next_question is None


def test_a_false_answer_does_not_count_as_satisfied():
    answers = {"standard_implemented": {"question_id": "standard_implemented", "satisfied": False, "detail": "no"}}
    state = compute_state(_app(answers=answers))
    assert state.all_satisfied is False
    assert state.next_question.id == "standard_implemented"
