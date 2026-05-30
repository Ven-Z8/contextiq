"""Cost estimation from token usage."""

from contextiq.llm.client import estimate_cost


def test_sonnet_cost():
    # 1M in @ $3, 1M out @ $15
    assert estimate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000) == 18.0


def test_opus_cost():
    assert estimate_cost("claude-opus-4-8", 1_000_000, 0) == 15.0


def test_haiku_cost():
    assert estimate_cost("claude-haiku-4-5", 0, 1_000_000) == 5.0


def test_small_usage_is_rounded():
    assert estimate_cost("claude-sonnet-4-6", 5000, 1000) == round(
        5000 / 1_000_000 * 3.0 + 1000 / 1_000_000 * 15.0, 6
    )


def test_unknown_model_returns_none():
    assert estimate_cost("gpt-4o", 1000, 1000) is None
