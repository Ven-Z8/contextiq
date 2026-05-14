# Wiki-Derived Patterns

Source vault: local AI engineering notes.

Implementation resource vault: local implementation notes and references.

These local notes should shape ContextIQ's implementation and portfolio narrative.

## ContextForge

- Context Window Auto-Compaction: summarize lower-value context instead of truncating blindly.
- Context-Minimization Pattern: include only context that earns its token cost.
- Progressive Disclosure for Large Files: retrieve document overview first, then sections, then exact pages/tables.
- Prompt Caching via Exact Prefix Preservation: keep static prompt prefixes identical across calls.
- Resource-Aware Optimization: token budget, cost, and latency are product behavior, not afterthoughts.

## RAGBench Pro

- Reliability problem map for RAG and agents.
- Workflow evals with mocked tools.
- Anti-reward-hacking grader design.
- Agent-augmented RAG should be benchmarked separately from static retrieval.

## DocIQ

- Define the ingestion contract before pipeline implementation.
- Preserve section and page context across every block.
- Use information extraction and Q&A prompts as explicit prompt templates, not inline strings.

## AgentOrchestra

- Planner-worker separation.
- Declarative multi-agent topology.
- Conditional parallel tool execution after the first linear flow works.
- Perception -> reason -> action as the internal shape of each worker.

## MCPForge and MCPGuard

- MCP is the shared tool gateway for document capabilities.
- Tool Capability Compartmentalization: one tool, one narrow capability.
- Policy-Gated Tool Proxy: validate access before every tool call.
- Hook-Based Safety Guard Rails: pre-tool and post-tool checks.
- Spec-first MCP schemas.

## PromptOpt

- APE-style compile loop: generate prompt candidates, score on held-out evals, keep winners.
- Self-critique evaluator loop.
- Spec-as-test feedback loop.

## EvalEngine

- Spec-driven eval contracts.
- Output verification loop.
- Incident-to-eval synthesis.
- Cost-per-task as a primary eval metric.
