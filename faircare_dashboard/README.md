# FairCare Research Results Viewer

A presentation-ready dashboard for the FairCare fairness-aware healthcare readmission prediction project.

## How to Open

Double-click `index.html` — no server needed.

## Dashboard Structure (10 Sections)

| # | Section | Content |
|---|---|---|
| 1 | **Overview** | Project goal, research question, class meaning |
| 2 | **Dataset Cleaning Pipeline** | Comprehensive 11-part explanation of data curation |
| 3 | **Experiment Design** | 4 experiments with strategies and research questions |
| 4 | **Architecture** | Mermaid diagram of full pipeline |
| 5 | **Research Results** | 16-model table, experiment cards, interpretation |
| 6 | **Experiment Viewer** | Full matrices with collapsible sections per model |
| 7 | **Fairness Findings** | Simple explanation cards + collapsed advanced tables |
| 8 | **Matrices Explained** | Plain-language guide to all 4 fairness matrices |
| 9 | **Guide** | Quick reference for matrix interpretation |
| 10 | **Next Steps** | Remaining work items |

## Dataset Cleaning Pipeline (Section 2)

This section has been extensively rewritten to explain the full data curation pipeline with real data metrics, including:

1. **Raw Dataset Used:** diabetic_data.csv dimensions and target distributions.
2. **Target Variable Preparation:** How '<30', '>30', and 'NO' were mapped to binary classes.
3. **Selected 18 Features Before Encoding:** Detailed table of features, grouped by type (Demographic, Diagnosis, etc.) and rationale.
4. **Cleaning Steps Applied:** Explicit steps traced back to Data_cleaning.ipynb.
5. **Missing and Unknown Value Handling:** Explanation of gender and race missing value drops.
6. **Encoding and Feature Engineering:** Exact mapping of categorical/ordinal values to model-ready features.
7. **Final Model-Ready Columns:** Details of the 39 final columns and the 81,474 x 39 shape of the clean baseline dataset.
8. **CSV Files Created:** Exact row/col shapes and origins of the 7 main CSVs.
9. **How Balanced Training Datasets Were Created:** Detailed experiment 001-004 source notebook and intent tracking.
10. **Dataset Population Comparison:** Cross-experiment tracking of Class 0 vs Class 1 distributions.
11. **Data Curation Diagram:** Mermaid diagram showing exactly which notebook produces which CSV.

## Fairness Findings Section (Section 7)

- 16-model table with Accuracy, Recall, FNR, F1, ROC-AUC and Simple Meaning.
- Four Simple Cards explaining Race, Gender, Age, and Overall fairness results.
- Collapsed Advanced Tables hiding detailed race/gender/age/gap numeric tables.
- Speaker script for presenting findings.

## Data Files

| File | Purpose |
|---|---|
| `index.html` | Generated dashboard |
| `style.css` | Dark theme with research components |
| `script.js` | Tabs, collapsibles, scroll navigation |
| `experiment_data.json` | Extracted notebook outputs |
| `fairness_gaps.json` | Computed recall/FNR/calibration gaps |
| `group_findings.json` | Per-group race/gender/age data |
| `README.md` | This file |

## Data Integrity

All values extracted directly from Jupyter notebook outputs and dataset CSVs. No fabricated values.
