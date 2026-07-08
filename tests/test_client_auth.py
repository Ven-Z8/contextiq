"""AnthropicLLMClient must neutralize an ambient ANTHROPIC_AUTH_TOKEN env var."""

from contextiq.llm.client import AnthropicLLMClient


def test_client_neutralizes_ambient_auth_token(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "polluting-token")
    c = AnthropicLLMClient(api_key="sk-ant-test", model="claude-sonnet-4-6")
    # No bearer Authorization header is emitted — x-api-key auth only.
    assert "Authorization" not in c.client.auth_headers
    assert c.client.auth_headers.get("X-Api-Key") == "sk-ant-test"
