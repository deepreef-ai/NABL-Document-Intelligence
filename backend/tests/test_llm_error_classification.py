"""Nova (Bedrock, boto3-based) must turn a botocore failure into the right
typed exception (so llm/chain.py's router can act on it) — see
llm/base.py's classify_boto_error."""
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from app.llm.base import (
    LlmAuthError,
    LlmBadRequestError,
    LlmProviderError,
    LlmQuotaError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from app.llm.providers import NovaProvider


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "Converse")


@pytest.mark.parametrize(
    "code,expected_exc",
    [
        ("ThrottlingException", LlmRateLimitError),
        ("TooManyRequestsException", LlmRateLimitError),
        ("AccessDeniedException", LlmAuthError),
        ("UnrecognizedClientException", LlmAuthError),
        ("ValidationException", LlmBadRequestError),
        ("ModelTimeoutException", LlmTimeoutError),
        ("InternalServerException", LlmProviderError),
    ],
)
def test_nova_classifies_bedrock_error_codes(monkeypatch, code, expected_exc):
    provider = NovaProvider("us.amazon.nova-2-lite-v1:0", "us-east-1", 30.0)

    class _FakeClient:
        def converse(self, **kwargs):
            raise _client_error(code)

    monkeypatch.setattr(provider, "_get_client", lambda: _FakeClient())

    with pytest.raises(expected_exc):
        provider.generate("system", "hello")


def test_nova_classifies_read_timeout():
    provider = NovaProvider("us.amazon.nova-2-lite-v1:0", "us-east-1", 30.0)

    class _FakeClient:
        def converse(self, **kwargs):
            raise ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")

    provider._get_client = lambda: _FakeClient()

    with pytest.raises(LlmTimeoutError):
        provider.generate("system", "hello")


def test_nova_sends_system_prompt_and_parses_converse_reply():
    provider = NovaProvider("us.amazon.nova-2-lite-v1:0", "us-east-1", 30.0)
    captured = {}

    class _FakeClient:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {"output": {"message": {"content": [{"text": "hi there"}]}}}

    provider._get_client = lambda: _FakeClient()

    result = provider.generate("be helpful", "hello")
    assert result == "hi there"
    assert captured["system"] == [{"text": "be helpful"}]
    assert captured["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]
    assert captured["modelId"] == "us.amazon.nova-2-lite-v1:0"


def test_nova_attaches_image_as_a_content_block():
    provider = NovaProvider("us.amazon.nova-2-lite-v1:0", "us-east-1", 30.0)
    captured = {}

    class _FakeClient:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

    provider._get_client = lambda: _FakeClient()

    provider.generate("system", "describe this", image=b"fake-bytes", image_media_type="image/jpeg")
    content = captured["messages"][0]["content"]
    assert content[0] == {"text": "describe this"}
    assert content[1] == {"image": {"format": "jpeg", "source": {"bytes": b"fake-bytes"}}}
