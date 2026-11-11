# PAMR–SARA: Metadata-Aware Email Retrieval Evaluation

This repository contains the analysis and data-preparation scripts for evaluating metadata-aware email retrieval on the SARA human-judged benchmark and the Berkeley Enron email collection.

## Repository contents

| File | Purpose |
|---|---|
| `README.md` | Project description, data sources, methods, and instructions |
| `Analysis Script.py` | End-to-end retrieval experiments, nested evaluation, statistical analysis, and result generation |
| `Data Processing.py` | Separate data acquisition, email parsing, benchmark preparation, validation, and processed-data export |

The raw datasets and generated processed-data files are **not** included in this repository. They are obtained or created when the scripts are run.

## Research objective

The analysis compares six email-retrieval systems using the original SARA human relevance judgments:

1. **BM25-body** — BM25 retrieval using email body text.
2. **TFIDF-body** — TF–IDF retrieval using email body text.
3. **BM25F-fielded** — field-aware BM25F using body, subject, sender, and recipient fields.
4. **Headers-only** — retrieval using the email metadata fields.
5. **BM25+headers** — body retrieval fused with weighted header scores.
6. **PAMR-availability** — body and header retrieval combined with an observed metadata-completeness feature.

The primary evaluation metric is NDCG@10. Secondary metrics are MAP, BPref, MRR, Precision@10, and Recall@100.

## Data sources

The scripts retrieve the original source files from the following public sources, using fallback URLs where available:

- **Berkeley Enron email collection:** [Berkeley Enron archive](https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz)
- **SARA human relevance judgments, query formulations, and information needs:** [SARA dataset on Zenodo](https://zenodo.org/records/8006820)
- **SARA GitHub fallback files:** [SARA repository](https://github.com/JackMcKechnie/SARA)

The analysis expects the original benchmark composition:

| Dataset component | Expected records |
|---|---:|
| Email documents | 1,702 |
| Query formulations | 150 |
| Independent information needs | 50 |
| Repeated human-judgment rows | 34,413 |
| Unique need–document assessments | 11,471 |

Each information need has three query formulations. The repeated relevance rows are retained in the source data, but the evaluation groups queries by information need so that the three formulations are not treated as independent information needs.

## Methods implemented in `Analysis Script.py`

### Retrieval and metadata processing

The analysis parses email body text and the `Subject`, `From`, and `To` fields. It also records whether selected headers and a parseable date are present. The metadata-completeness score is the proportion of five observed fields that are present: sender, recipient, subject, Message-ID, and parseable date.

The completeness score describes observed header availability only. It is **not** an authentication measure and does not establish sender identity, document authenticity, source trustworthiness, or information provenance.

### Model selection and evaluation

- Five-fold outer cross-validation is grouped by information need.
- Four-fold inner grouped cross-validation selects model parameters using training information needs only.
- The primary metric is NDCG@10; the script also calculates MAP, BPref, MRR, Precision@10, and Recall@100.
- Confidence intervals are estimated by resampling information needs, rather than treating the 150 query formulations as independent units.
- Paired Wilcoxon comparisons and Holm adjustment are applied to the primary metric comparisons against BM25-body.
- Metadata-field ablations, controlled metadata-masking experiments, and exploratory query-latency measurements are also generated.
- A fixed random seed is used for reproducibility.

The script reports measured outcomes, including results that do not favour PAMR. It does not assume that the proposed method outperforms the baseline.

## How to run

Python 3.10 or later is recommended. The scripts install their required Python packages if they are not already available.

### 1. Run the separate data-processing script

In Google Colab, upload `Data Processing.py` and run:

```bash
!python "/content/Data Processing.py"
```

Or, from a terminal in the repository directory:

```bash
python "Data Processing.py"
```

By default, the script downloads the original source files into `data/raw/`, validates them, and creates:

```text
processed_data/
├── emails_processed.csv
├── queries_processed.csv
├── information_needs_processed.csv
├── human_qrels_processed.csv
├── unique_need_document_judgments.csv
├── metadata_field_coverage.csv
├── data_processing_report.json
└── source_manifest.json

processed_data.zip
```

The processed email CSV contains email body text and header values. Keep generated data files local and do not commit them to GitHub unless you have reviewed the applicable dataset terms and data-protection implications.

You can provide already-downloaded source files instead of downloading them again:

```bash
python "Data Processing.py" \
  --email-archive "path/to/enron_with_categories.tar.gz" \
  --qrels "path/to/repeated_qrels.txt" \
  --queries "path/to/repeated_queries.tsv" \
  --needs "path/to/information_needs.tsv"
```

To use files already present in `data/raw/` without attempting downloads:

```bash
python "Data Processing.py" --no-download
```

### 2. Run the complete analysis

In Google Colab, upload `Analysis Script.py` and run:

```bash
!python "/content/Analysis Script.py"
```

Or, from a terminal in the repository directory:

```bash
python "Analysis Script.py"
```

The analysis script independently retrieves and processes the original source data, applies its own strict validation, runs the retrieval experiments, and creates:

```text
PAMR_SARA_ICDE2027_RESULTS/
PAMR_SARA_ICDE2027_RESULTS.zip
```

The analysis script does not require the CSV files generated by `Data Processing.py`. The separate data-processing script is provided to make source preparation, intermediate datasets, and validation outputs independently inspectable and reproducible.

### Software self-test

The analysis script supports a software-only test mode:

```bash
python "Analysis Script.py" --self-test
```

This mode uses a synthetic fixture to test the software workflow. **Its output is not original-dataset evidence and must not be reported as research results.**

## Generated analysis outputs

The complete analysis creates the following output areas:

- `tables/` — CSV and Markdown result tables.
- `figures/` — publication-resolution PNG and PDF figures.
- `runs/` — held-out metrics, rankings, ablations, metadata analyses, and timing outputs.
- `audit/` — validation results, source hashes, selected parameters, and experiment manifest.
- `CORE_RESULTS_SUMMARY.md` — automatically generated summary of the primary held-out findings.
- `README_FIRST.md` — description of the generated experiment package.

The script also checks the generated results archive for ZIP integrity before reporting successful completion.

## Reproducibility and interpretation notes

- The original human relevance judgments are used; the scripts do not generate or replace human labels.
- Repeated judgments across the three formulations of an information need are not treated as independent information needs.
- Parameter selection is restricted to inner training folds; held-out outer folds are used for evaluation.
- The data-processing and analysis scripts validate the expected document, query, information-need, and judgment counts, along with document-ID alignment and repeated-judgment consistency.
- Observed metadata completeness must not be interpreted as verified provenance.
- Runtime measurements are exploratory measurements on the specified collection and do not establish enterprise-scale scalability.
- The public corpus and benchmark remain subject to their original terms, licensing conditions, and applicable ethical/data-protection requirements.

#
