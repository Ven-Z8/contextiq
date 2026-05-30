from contextiq.query import answer_question


def test_answer_question_returns_pair():
    answer, packet = answer_question("What are the termination obligations?")
    assert hasattr(answer, "text")
    assert hasattr(packet, "question")
