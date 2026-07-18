"""Simple agentic retrieval: decompose -> retrieve + merge -> rerank.

Targets the measured failure mode (financial tables / line-items not retrieved).
Decomposition rewrites a concept question ("quick ratio") into the underlying
line items ("current assets", "current liabilities", "cash and cash equivalents")
so hybrid search can pull the balance-sheet table. A table-aware LLM rerank then
keeps the most useful blocks (including tables) within the answer's token budget.

No multi-agent orchestration — one decompose call, one rerank call. Built
alongside plain hybrid_hits so it can be measured head-to-head before adoption.
"""

from __future__ import annotations

import json
import logging
import re

from contextiq.llm.client import LLMClient
from contextiq.retrieval.models import RetrievalHit
from contextiq.retrieval.store import LocalDocumentStore

logger = logging.getLogger(__name__)

ROUTER_SYSTEM = (
    "You pick which filing a question is about. Given the question and a numbered list of "
    "filing identifiers (e.g. AMD_2022_10K-...), reply with ONLY the number of the single "
    "filing the question targets, or -1 if it is unclear. Output just the number."
)

DECOMPOSE_SYSTEM = (
    "You turn a financial question about a 10-K into focused search queries. "
    "Output ONLY a JSON array of 5-7 short query strings. For revenue questions, "
    "include SPECIFIC income statement line items from the revenue section "
    "(e.g. for American Express: 'discount revenue', 'net card fees', 'other fees and commissions', "
    "'total non-interest revenues', 'interest income', 'interest on loans', 'total interest income'). "
    "ALSO include this exact pattern for revenue tables: "
    "'Year Ended December 31 (Millions) Revenues Discount revenue Net card fees'. "
    "IMPORTANT: Match the company's actual terminology - use 'Total revenue' or 'Net revenue' or 'Net sales' "
    "as appears in their consolidated statement of income. "
    "For ratio questions, include the underlying balance sheet line items "
    "(e.g. quick ratio -> 'cash and cash equivalents', 'short-term investments', "
    "'accounts receivable', 'current liabilities', 'current assets'). "
    "For segment questions, include known segment names if possible. "
    "IMPORTANT: Also include the EXACT financial statement header text like "
    "'Net revenue for 2022 was $23.6 billion' or 'total revenues net of interest expense $52,862 million' "
    "to match the narrative text around tables. No prose, only the JSON array."
)

RERANK_SYSTEM = (
    "You rank retrieved passages by how useful they are for answering the financial question. "
    "CRITICAL: Strongly prefer passages that are TABLES containing specific dollar amounts, "
    "revenue figures, line items from income statement, balance sheet, or cash flow statement. "
    "Prefer tables with columns for multiple years (2022, 2021, 2020). "
    "Deprioritize generic text, legal disclaimers, risk factors, and executive compensation tables. "
    "Output ONLY a JSON array of the passage numbers (integers) in best-first order."
)

# Few-shot examples for reranking
RERANK_EXAMPLES = '''
Example 1:
Question: "What is American Express revenue?"
Passages:
[0] (text) "Our spend-centric business model focuses on generating revenues primarily by driving spending on our cards..."
[1] (table) | Year Ended Dec 31 (Millions) | 2022 | 2021 | 2020 | | Revenues | 52,862 | 43,661 | 36,109 | | Discount revenue | 25,727 | 20,401 | 26,167 |
[2] (text) "Delta Air Lines is our largest strategic partner..."
Output: [1, 0, 2]

Example 2:
Question: "What is AMD quick ratio?"
Passages:
[0] (text) "AMD maintains strong liquidity position..."
[1] (table) | | 2022 | 2021 | | Cash and cash equivalents | 2,366 | 3,219 | | Short-term investments | 1,582 | 2,105 | | Accounts receivable | 299 | 231 | | Current liabilities | 2,140 | 1,980 |
[2] (text) "The company's cash flow from operations was..."
Output: [1, 2, 0]

Example 3:
Question: "What is AES segment revenue?"
Passages:
[0] (table) | Year Ended Dec 31 (Millions) | 2022 | 2021 | 2020 | | MCAC SBU revenue | 4,123 | 3,891 | 3,456 | | South America SBU revenue | 2,987 | 2,765 | 2,432 | | US & Utilities SBU revenue | 3,567 | 3,234 | 2,891 |
[1] (text) "AES operates through four strategic business units..."
[2] (table) | Executive compensation | 2022 | 2021 | | CEO salary | 1,200 | 1,150 |
Output: [0, 1, 2]
'''


def _extract_json_array(text: str) -> str:
    """minimax/nemotron may wrap JSON in reasoning/markdown — grab the outermost [...] span."""
    import re
    # Strip common reasoning/markdown wrappers
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start, end = text.find("["), text.rfind("]")
    return text[start : end + 1] if start != -1 and end > start else "[]"


def decompose(client: LLMClient, question: str) -> list[str]:
    """Return 2-4 sub-queries (line items for ratios/metrics). [] on parse failure."""
    result = client.generate(
        system_prompt=DECOMPOSE_SYSTEM, user_prompt=question, max_tokens=400
    )
    try:
        parsed = json.loads(_extract_json_array(result.text))
        return [s.strip() for s in parsed if isinstance(s, str) and s.strip()][:4]
    except Exception as exc:
        logger.warning("Decompose parse failed; using original query only", exc_info=exc)
        return []


def _rerank(
    client: LLMClient, question: str, candidates: list[RetrievalHit], k: int
) -> list[RetrievalHit]:
    listing = "\n".join(
        f"[{i}] ({c.block.block_type.value}) {c.block.text[:500]}"
        for i, c in enumerate(candidates)
    )
    result = client.generate(
        system_prompt=RERANK_SYSTEM,
        user_prompt=f"{RERANK_EXAMPLES}\n\nQuestion: {question}\n\nPassages:\n{listing}",
        max_tokens=1000,
    )
    try:
        order = [int(x) for x in json.loads(_extract_json_array(result.text))]
    except Exception:
        order = []
    seen: set[int] = set()
    ranked: list[RetrievalHit] = []
    for idx in order:
        if 0 <= idx < len(candidates) and idx not in seen:
            ranked.append(candidates[idx])
            seen.add(idx)
    # Append any the model didn't rank so nothing is silently dropped before the cut.
    for i, candidate in enumerate(candidates):
        if i not in seen:
            ranked.append(candidate)
    return ranked[:k]


def _parse_index(text: str, n: int) -> int:
    """Last integer in the model's reply, validated as a 0..n-1 index (else -1)."""
    ints = re.findall(r"-?\d+", text)
    for token in reversed(ints):
        value = int(token)
        if -1 <= value < n:
            return value
    return -1


def _normalize_company_name(name: str) -> str:
    """Normalize company name for matching against document IDs."""
    # Replace common variations
    normalized = name.lower()
    normalized = normalized.replace("&", "and")
    normalized = normalized.replace(".", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("'", "")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("_", " ")
    # Collapse whitespace
    normalized = " ".join(normalized.split())
    return normalized


def route_document(store: LocalDocumentStore, question: str, client: LLMClient) -> str | None:
    """Pick the ingested filing the question targets, so retrieval can scope to it.

    Corpus disambiguation: "current assets / current liabilities" matches every
    company's balance sheet, so without routing a cross-company corpus returns a
    mix. Returns a document_id, or None when there is nothing to route to.
    """
    docs = sorted(store._load_manifest())
    if len(docs) <= 1:
        return docs[0] if docs else None
    listing = "\n".join(f"[{i}] {doc}" for i, doc in enumerate(docs))
    result = client.generate(
        system_prompt=ROUTER_SYSTEM,
        user_prompt=f"Question: {question}\n\nFilings:\n{listing}",
        max_tokens=500,
    )
    idx = _parse_index(result.text, len(docs))
    if idx >= 0:
        return docs[idx]
    
    # Fallback: fuzzy match company name from question to document_id
    question_normalized = _normalize_company_name(question)
    for doc in docs:
        # Extract company prefix (e.g., "COMPANY" from "COMPANY_2016_10K-...")
        company = doc.split('_')[0].lower()
        if company in question_normalized:
            return doc
    
    # LLM-assisted normalization fallback: ask LLM which filing matches
    # This handles cases like "Johnson & Johnson" -> "JOHNSON_JOHNSON"
    fallback_prompt = (
        f"The user asked: '{question}'\n\n"
        f"Available filings: {', '.join(docs[:20])}{'...' if len(docs) > 20 else ''}\n\n"
        f"Which filing does the question refer to? Reply with ONLY the exact document_id "
        f"from the list above, or -1 if unclear."
    )
    fallback_result = client.generate(
        system_prompt="You match a user question to the correct financial filing. Output only the document_id or -1.",
        user_prompt=fallback_prompt,
        max_tokens=100,
    )
    fallback_idx = _parse_index(fallback_result.text, len(docs))
    if 0 <= fallback_idx < len(docs):
        return docs[fallback_idx]
    
    return None


def agentic_retrieve(
    store: LocalDocumentStore,
    question: str,
    client: LLMClient,
    k: int = 15,
    per_query: int = 25,
) -> list[RetrievalHit]:
    """Route to the target filing (if unscoped) -> decompose -> retrieve/merge -> rerank."""
    if store._scoped_document_id is None:
        target = route_document(store, question, client)
        if target is not None:
            store = store.scoped(target)

    subqueries = [question, *decompose(client, question)]

    pool: dict[str, RetrievalHit] = {}
    for subquery in subqueries:
        for hit in store.hybrid_hits(subquery, limit=per_query, group_by_section=False):
            existing = pool.get(hit.block.block_id)
            if existing is None or hit.score > existing.score:
                pool[hit.block.block_id] = hit

    candidates = list(pool.values())
    if len(candidates) <= k:
        return candidates
    # Apply LLM rerank to boost table/financial blocks
    return _rerank(client, question, candidates, k)
