# Document Gauntlet

The first portfolio demo should use documents that are hard enough to expose weak RAG systems.

## Primary Documents

- Apple 2025 Form 10-K: https://investor.apple.com/sec-filings/sec-filings-details/default.aspx?FilingId=18880179
- Microsoft 2025 Annual Report and 10-K: https://www.microsoft.com/investor/reports/ar25/download-center/index.html
- NVIDIA Annual Reports: https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx

## Visual / Graph-Heavy Documents

- Apple Environmental Progress Reports: https://www.apple.com/environment/
  - Best for polished charts, product visuals, emissions metrics, recycling/materials claims, and figure-caption retrieval.
- NVIDIA Annual Reports: https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx
  - Best for large visual annual reports, business graphics, financial tables, and AI/platform narrative.
- NASA Data and Computing Architecture Study: https://science.nasa.gov/wp-content/uploads/2024/08/data-and-computing-architecture-study-final-report-aug-2024.pdf
  - Best for architecture diagrams, pie charts, layered service models, and technical recommendations.
- NASA Moon to Mars Architecture Definition Documents: https://www.nasa.gov/moontomarsarchitecture-architecturedefinitiondocuments/
  - Best for complex systems architecture, mission diagrams, mapping tables, and technical PDF stress tests.
  - The objective mapping downloads may open as Apple Numbers on macOS. Export them as `.xlsx` before ingestion; ContextIQ reads Excel workbooks directly with `openpyxl`.
- IPCC AR6 Synthesis Report Summary for Policymakers: https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf
  - Best for dense scientific figures, multi-panel climate charts, confidence language, and evidence-heavy visual explanations.
- IRENA World Energy Transitions Outlook 2024: https://www.irena.org/Publications/2024/Nov/World-Energy-Transitions-Outlook-2024
  - Best for energy-transition charts, scenario comparisons, regional data, and graph/table retrieval.

## Legal and Regulatory Documents

- FTC v. Amazon case materials: https://www.ftc.gov/legal-library/browse/cases-proceedings/1910129-1910130-amazoncom-inc-amazon-ecommerce
- EU AI Act: https://eur-lex.europa.eu/legal-content/EN/TXT/?qid=1721811182344&uri=CELEX%3A32024R1689
- NIST AI RMF: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10
- NIST Generative AI Profile: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence

## Contract Exhibits

- Richtech Robotics master services agreement exhibit: https://www.sec.gov/Archives/edgar/data/1963685/000121390025080120/0001213900-25-080120-index.html

## Stress Questions

- What are the top risks and what evidence supports each risk?
- Which claims are financial facts versus management narrative?
- Extract term, termination, confidentiality, IP ownership, and payment obligations.
- Cite every claim with source, page, and section.
- What evidence is weak, missing, or ambiguous?
- Find the chart/figure that explains the architecture, emissions trend, or scenario pathway.
- Explain what the retrieved chart shows using only its caption, Docling visual
  description, surrounding section, table text, and page citation.
- Which visual evidence supports the claim, and did it come from vector, lexical, section anchor, or expansion retrieval?
