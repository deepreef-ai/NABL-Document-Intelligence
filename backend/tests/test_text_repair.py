from app.dataset_normalization.text_repair import repair_glued_words


def test_repair_glued_words_splits_a_long_unbroken_narrative_run():
    text = "Thisistocertifythattheconsignmentcomplieswiththefollowingspecification"
    result = repair_glued_words(text)
    assert result == "This is to certify that the consignment complies with the following specification"


def test_repair_glued_words_leaves_normal_spaced_text_untouched():
    text = "Patient Name: John Doe, Age: 45"
    assert repair_glued_words(text) == text


def test_repair_glued_words_leaves_short_runs_alone_to_avoid_mangling_real_words():
    # "Histopathology" (14 chars) is a real single word — below the
    # unambiguous-multi-word threshold, it must be left exactly as-is
    # rather than risk splitting a legitimate long technical term.
    text = "DepartmentHistopathology"[:14]
    assert repair_glued_words(text) == text


def test_repair_glued_words_only_touches_the_glued_run_not_the_rest_of_the_line():
    text = "Date:14Nov2016 Thisistocertifythattheconsignmentcomplieswiththefollowingspecification end."
    result = repair_glued_words(text)
    assert result.startswith("Date:14Nov2016 ")
    assert result.endswith(" end.")
    assert "This is to certify" in result


def test_repair_glued_words_handles_empty_and_none_like_input():
    assert repair_glued_words("") == ""
