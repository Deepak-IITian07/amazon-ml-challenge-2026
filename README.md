# Business Entity Resolution Pipeline

Amazon ML Challenge 2026

## 1. Overview
This package provides the complete, self-contained, reproducible pipeline for the Business Entity Resolution Challenge.
Given noisy entity records across 3 independent sources (`Source 1`, `Source 2`, and `Source 3`), the pipeline maps each `Source 1` reference entity to all corresponding matching entities from `Source 2` and `Source 3`.

Outputs generated:
1. `output/matching_results.tsv` — Scored final predictions (`source1_entity_id`, `matched_entity_ids`).
2. `output/candidate_pairs.tsv` — Blocking candidate pairs fed to the classifier (`source1_entity_id`, `candidate_entity_ids`).

---

## 2. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── preprocessing.py    # Multi-script transliteration, legal suffix stripping, address parsing
│   ├── blocking.py         # Multi-key weighted inverted index blocking engine
│   ├── features.py         # 13-dimensional pairwise lexical, fuzzy, and structural features
│   ├── model.py            # LightGBM classifier with F_0.5 decision threshold calibration
│   ├── metrics.py          # Official Macro F_0.5 evaluation metric computation
│   └── pipeline.py         # End-to-end streaming orchestrator
├── model_lgb.txt           # Trained LightGBM booster weights
├── requirements.txt        # Pinned dependencies
└── README.md               # Reproduction instructions
```

---

## 3. Installation & Environment Setup

Python 3.10+ is supported. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Execution Guide

All commands should be executed from the `student_resource/` directory.

### Step 1: Model Training & Threshold Calibration (Optional if using pre-trained model)
To train the LightGBM matching model on the training dataset and calibrate the decision threshold for $F_{0.5}$:

```bash
python3 code/business_entity_resolution/src/pipeline.py \
    --mode train \
    --train-dir dataset/train \
    --model-path code/business_entity_resolution/model_lgb.txt \
    --sample-entities 5000
```

### Step 2: Test Set Inference
To run streaming candidate generation and matching inference across the complete test set (`France`, `US`, `India`):

```bash
python3 code/business_entity_resolution/src/pipeline.py \
    --mode infer \
    --test-dir dataset/test \
    --output-dir output \
    --model-path code/business_entity_resolution/model_lgb.txt \
    --threshold 0.40
```

### Step 3: Submission Format Validation
Validate both generated TSV files against competition rules:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
A return code of `0` (`PASS`) confirms full compliance with submission formatting standards.
