def test_init_obs_is_callable():
    from contextiq.obs import init_obs

    init_obs(enabled=False)  # no-op must not raise


def test_answer_question_emits_run_span(monkeypatch):
    # Wire an in-memory exporter as the global provider
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

    # Stub the heavy pipeline so the test needs no Qdrant/LLM
    import contextiq.query as q

    class _Pkt:
        question = "why?"
        sources = []
        token_budget = 6000
        used_tokens = 100
        dropped_candidates = 2

    class _Ans:
        text = "a"; model = "claude-sonnet-4-6"; mode = "llm"
        tokens_in = 10; tokens_out = 5; cost_usd = 0.01; warnings = []

    monkeypatch.setattr(q, "_build_packet", lambda question: _Pkt(), raising=False)
    monkeypatch.setattr(q, "_synthesize", lambda packet: _Ans(), raising=False)

    ans, pkt = q.answer_question("why?")
    from ven_obs import api
    spans = {s.name: dict(s.attributes or {}) for s in exporter.get_finished_spans()}
    assert "contextiq.run" in spans
    run = spans["contextiq.run"]
    assert run[api.ATTR_PROJECT] == "contextiq"
    assert run[api.ATTR_MODEL] == "claude-sonnet-4-6"
    import json
    assert json.loads(run[api.ATTR_EXTRA])["citations_count"] == 0
