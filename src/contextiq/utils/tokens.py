"""Token counting helpers."""

from __future__ import annotations

from collections.abc import Sequence

import tiktoken


class TokenCounter:
    """Count tokens with tiktoken, falling back to a local estimate offline."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            self.encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoder = None

    def count(self, text: str) -> int:
        """Return a token count for text."""

        if self.encoder is not None:
            return len(self.encoder.encode(text))
        return max(1, len(text.split()) + len(text) // 20)

    def encode(self, text: str) -> Sequence[int]:
        """Return token ids where available, or estimate-sized placeholders."""

        if self.encoder is not None:
            return self.encoder.encode(text)
        return range(self.count(text))
