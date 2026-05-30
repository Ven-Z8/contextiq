from contextiq.query import answer_question


def test_answer_question_returns_pair():
    answer, packet = answer_question("What are the termination obligations?")
    assert hasattr(answer, "text")
    assert hasattr(packet, "question")


def test_provider_shape():
    from evals.provider import ContextiqProvider

    resp = ContextiqProvider().call_api("What are the termination obligations?", {}, {})
    assert isinstance(resp["output"], str)
    assert "metadata" in resp
