# ContextIQ Architecture

```text
PDF / HTML / SEC exhibit
        |
        v
DocIQ Ingestion
Docling -> DocumentBlock[]
DocumentChunker -> retrieval-ready chunks
        |
        v
Qdrant Retrieval
query -> candidates with metadata + trace
        |
        v
ContextForge
rank -> budget -> pack -> quality report
        |
        v
Anthropic LLM / Agent workflow
answer -> citations -> verifier -> eval
```

## Multimodal Evidence

Image and graph retrieval does not require blind LLM search. Docling emits tables
as markdown blocks and figures as caption text blocks with visual metadata. For
PDFs, ContextIQ enables Docling page/picture image generation plus best-effort
picture classification and picture description. Stored figure metadata includes
captions, Docling visual descriptions, visual class predictions, image artifact
paths, page numbers, and bounding boxes when available. If local picture
enrichment fails, ingestion retries with the basic Docling visual pipeline.

The LLM layer may explain retrieved visual evidence with citations, but the
answer prompt treats Docling visual descriptions as supporting evidence rather
than exact chart-value ground truth unless numeric values are present in the
retrieved text or table rows.

## Chunking Policy

ContextIQ chunks after parsing and before storage/indexing. This follows the
Qdrant principle that focused chunks retrieve better than whole-document or
giant-table embeddings.

- Text: split oversized prose into overlapping word windows while preserving
  section/page metadata.
- Tables and spreadsheets: split oversized markdown tables into overlapping
  row windows and repeat the header in every chunk.
- Figures: keep atomic and index caption text, Docling visual descriptions, and
  visual metadata.

Every chunk carries `parent_block_id`, `chunk_index`, `chunk_count`, strategy,
and row/word range metadata so the UI and later answer layer can explain why a
piece of evidence was retrieved.

## Retrieval Profile

Generic retrieval code reads corpus vocabulary from
`config/retrieval_profile.json`. This profile contains source aliases, document
title aliases, asset modes, known assets, product/service markers, financial
metrics, and structured-document source markers. Adding Tesla filings, legal
contracts, NIST reports, or new NASA workbooks should be a profile update first,
not a ranker code edit.
