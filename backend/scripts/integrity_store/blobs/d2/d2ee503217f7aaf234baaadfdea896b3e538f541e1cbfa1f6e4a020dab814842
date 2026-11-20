from app.documents import verification


class FakeChain:
    def __init__(self, reply_fields):
        self.reply_fields = reply_fields
        self.calls = 0

    def generate_json(self, system, user_text):
        self.calls += 1
        return {"fields": self.reply_fields}


def test_verify_fields_asks_about_every_field_in_one_call_and_normalizes_the_reply(monkeypatch):
    fake_chain = FakeChain([
        {"field": "organisation.pan_number", "value": "ABCDE1234F"},  # confidence omitted, like qwen2.5:3b does
        {"field": "organisation.tan_number", "value": "DELP12345E", "confidence": 0.8},
    ])
    monkeypatch.setattr(verification, "get_chunked_extraction_chain", lambda: fake_chain)

    result = verification.verify_fields(
        ["organisation.pan_number", "organisation.tan_number", "organisation.telephone"],
        "some source text",
    )

    assert fake_chain.calls == 1
    by_field = {f["field"]: f for f in result}
    assert by_field["organisation.pan_number"]["confidence"] == 0.5  # defaulted, not a KeyError
    assert by_field["organisation.tan_number"]["confidence"] == 0.8
    # A field the model didn't mention (organisation.telephone) is simply
    # absent from the reply, not backfilled here — the retry pass only cares
    # about fields it actually found on this second attempt.
    assert "organisation.telephone" not in by_field


def test_verify_fields_prompt_lists_every_requested_field(monkeypatch):
    captured = {}

    class CapturingChain:
        def generate_json(self, system, user_text):
            captured["user_text"] = user_text
            captured["system"] = system
            return {"fields": []}

    monkeypatch.setattr(verification, "get_chunked_extraction_chain", lambda: CapturingChain())

    verification.verify_fields(["a.b", "c.d"], "source text here")

    assert "a.b" in captured["user_text"]
    assert "c.d" in captured["user_text"]
    assert "source text here" in captured["user_text"]
