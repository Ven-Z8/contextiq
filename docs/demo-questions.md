# Demo Questions

Active corpus for this phase:

- Apple 2025 Form 10-K
- NASA Moon to Mars lunar objective decomposition workbook
- Microsoft FY25 Q4 Form 10-K
- NVIDIA 2025 Annual Report
- NASA Science Mission Directorate Data and Computing Architecture Study

Keep the DB clean: only real external documents belong in the active vector store. Research notes and planning notes should stay in the Obsidian vault, not in the demo corpus.

| # | Question | Expected Evidence | Skill Shown |
|---:|---|---|---|
| 1 | What is the Products and Services Performance in 2025? | Apple page 26 net-sales-by-category table and adjacent MD&A explanations. | SEC table retrieval and grounded financial synthesis |
| 2 | What was Microsoft revenue, gross margin, operating income, and net income in fiscal 2025? | Microsoft income table showing revenue $281,724M, gross margin $193,893M, operating income $128,528M, and net income $101,832M. | Named-company source precision over broad financial language |
| 3 | What were NVIDIA fiscal 2025 revenue, gross margin, operating income, and diluted earnings per share? | NVIDIA results table showing revenue $130,497M, gross margin 75.0%, operating income $81,453M, and diluted EPS $2.94. | Cross-filing table retrieval without Apple/Microsoft bleed |
| 4 | What did the NASA Data and Computing Architecture Study recommend? | NASA study recommendation blocks for a core architecture, hybrid cloud/HEC, cybersecurity support, and programmatic alignment. | Report-title precision and recommendation retrieval |
| 5 | What is UC ID UC-T-202 L? | NASA rows mapping UC-T-202 L to transportation of large cargo from Earth to the lunar surface. | Exact-code spreadsheet lookup |
| 6 | What HLR functions support Orion? | NASA asset-function mapping rows containing Orion and EM-003-HLR. | Asset/function table retrieval |
| 7 | Which NASA functions and use cases describe lunar power generation and distribution? | NASA power rows with FN-P-101, FN-P-301, UC-P-101, and UC-P-301. | Multi-row structured spreadsheet retrieval |
| 8 | What NASA needs describe continuous power for crew safety critical operations? | NASA CN-P-103 and related continuous power / mission critical activity rows. | Need/use-case retrieval over wide tables |
| 9 | How did Products and Services gross margin perform in 2025? | Apple page 27 gross margin table and Products/Services explanations. | Multi-table financial context assembly |
| 10 | What tariff impacts did Apple describe in 2025? | Apple tariff risk language and Products gross margin tariff-cost explanation. | Risk + MD&A evidence linking |
| 11 | What does Microsoft say about service and other revenue in 2025? | Microsoft revenue table separating Product and Service and other revenue. | Segment-like table extraction from DOCX |
| 12 | Which NVIDIA table summarizes fiscal 2025 results? | NVIDIA fiscal 2025 results table on revenue, gross margin, operating income, and diluted EPS. | Visual/table-heavy annual-report retrieval |
