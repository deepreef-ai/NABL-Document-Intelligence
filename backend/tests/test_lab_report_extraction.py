from app.documents.app import extract_lab_report


class FakeChain:
    """Stands in for LlmChain — records what it was called with and returns
    a canned reply, so these tests cover extract_lab_report's own
    normalization without any network/Bedrock call."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_json(self, system, user_text, image=None, image_media_type=None):
        self.calls.append({"system": system, "user_text": user_text, "image": image, "image_media_type": image_media_type})
        return self.reply


def test_confidence_is_returned_per_field_and_per_test_row():
    chain = FakeChain({
        "fields": {"report_number": "R-123", "client_name": "Acme Labs"},
        "field_confidence": {"report_number": 0.95, "client_name": 0.6},
        "tests": [{"test_name": "pH", "result": "6.5", "confidence": 0.9}],
    })
    result = extract_lab_report(chain, "some ocr text")

    assert result["fields"] == {"report_number": "R-123", "client_name": "Acme Labs"}
    assert result["field_confidence"] == {"report_number": 0.95, "client_name": 0.6}
    assert result["tests"][0]["confidence"] == 0.9


def test_field_returned_without_a_confidence_defaults_to_uncertain():
    """A value the model gave but skipped the confidence for is uncertain
    (0.5), not absent — dropping the field would lose real extracted data."""
    chain = FakeChain({
        "fields": {"report_number": "R-123", "client_name": "Acme Labs"},
        "field_confidence": {"report_number": 0.95},  # client_name omitted
        "tests": [{"test_name": "pH", "result": "6.5"}],  # no confidence at all
    })
    result = extract_lab_report(chain, "text")

    assert result["field_confidence"] == {"report_number": 0.95, "client_name": 0.5}
    assert result["tests"][0]["confidence"] == 0.5
    assert result["tests"][0]["result"] == "6.5"  # the row's real data is untouched


def test_confidence_entry_for_a_field_that_was_not_returned_is_dropped():
    chain = FakeChain({
        "fields": {"report_number": "R-123"},
        "field_confidence": {"report_number": 0.9, "never_extracted": 0.8},
    })
    result = extract_lab_report(chain, "text")
    assert result["field_confidence"] == {"report_number": 0.9}


def test_malformed_confidence_values_fall_back_to_uncertain():
    chain = FakeChain({
        "fields": {"a": "1", "b": "2", "c": "3", "d": "4"},
        "field_confidence": {"a": "high", "b": 7.5, "c": None, "d": True},
    })
    result = extract_lab_report(chain, "text")
    # A string, an out-of-range number, null, and a bool are all unusable.
    assert result["field_confidence"] == {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}


def test_malformed_reply_shapes_never_raise():
    empty = {"fields": {}, "field_confidence": {}, "field_verified": {}, "tests": []}
    assert extract_lab_report(FakeChain("not a dict at all"), "text") == empty
    assert extract_lab_report(FakeChain({"fields": ["wrong", "type"], "tests": "also wrong"}), "text") == empty
    # A non-dict entry inside an otherwise valid tests list is skipped, not fatal.
    result = extract_lab_report(FakeChain({"tests": ["junk", {"test_name": "pH", "result": "6.5"}]}), "text")
    assert len(result["tests"]) == 1
    assert result["tests"][0]["test_name"] == "pH"


def test_values_found_in_the_ocr_text_are_verified_and_invented_ones_are_not():
    """The strongest trust signal we have: a value absent from the OCR text
    we sent was inferred, not read (measured 34/34 such values wrong)."""
    chain = FakeChain({
        "fields": {"report_number": "R-123", "lab_name": "Invented Labs"},
        "tests": [
            {"test_name": "pH", "result": "6.5"},
            {"test_name": "Fat", "result": "99.9"},  # not in the source text
        ],
    })
    result = extract_lab_report(chain, "Report Number: R-123\npH 6.5 units")

    assert result["field_verified"] == {"report_number": True, "lab_name": False}
    assert result["tests"][0]["result_verified"] is True
    assert result["tests"][1]["result_verified"] is False


def test_verification_ignores_spacing_case_and_dash_differences():
    """OCR writes "40 - 129" where the model returns "40-129" — neither is
    an error, so a raw substring check would produce false alarms."""
    chain = FakeChain({
        "fields": {"range": "40-129", "date": "02Nov2020", "name": "acme labs"},
    })
    result = extract_lab_report(chain, "Bio Ref Interval 40 – 129\nDrawn 02 Nov 2020\nACME LABS")
    assert result["field_verified"] == {"range": True, "date": True, "name": True}


def test_nothing_is_flagged_when_there_is_no_source_text_to_check_against():
    """"Couldn't check" is not the same claim as "invented" — flagging
    everything would bury the real signal in noise."""
    chain = FakeChain({"fields": {"a": "1"}, "tests": [{"test_name": "pH", "result": "6.5"}]})
    result = extract_lab_report(chain, "   ")
    assert result["field_verified"] == {}
    assert "result_verified" not in result["tests"][0]


def test_empty_and_null_values_are_never_counted_as_verified():
    chain = FakeChain({
        "fields": {"blank": "", "nothing": None},
        "tests": [{"test_name": "pH", "result": None}],
    })
    result = extract_lab_report(chain, "some real source text")
    assert result["field_verified"] == {"blank": False, "nothing": False}
    assert result["tests"][0]["result_verified"] is False


def test_image_is_passed_through_to_the_chain_when_given():
    chain = FakeChain({"fields": {}, "tests": []})
    extract_lab_report(chain, "text", image=b"pngbytes", image_media_type="image/png")
    assert chain.calls[0]["image"] == b"pngbytes"
    assert chain.calls[0]["image_media_type"] == "image/png"


def test_text_only_call_passes_no_image():
    chain = FakeChain({"fields": {}, "tests": []})
    extract_lab_report(chain, "text")
    assert chain.calls[0]["image"] is None
    assert chain.calls[0]["image_media_type"] is None
