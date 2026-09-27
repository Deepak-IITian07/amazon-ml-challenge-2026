"""
End-to-End Orchestrator Pipeline for Business Entity Resolution
Handles country-sharded candidate blocking, feature extraction, model inference,
and streaming generation of matching_results.tsv and candidate_pairs.tsv.
"""

import os
import sys
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

# Ensure src modules are resolvable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import extract_brand_stem, extract_address_components
from src.blocking import InvertedIndexBlocker
from src.features import extract_pairwise_features
from src.model import ERMatchingModel
from src.metrics import evaluate_macro_f05


def run_country_inference(
    country: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    model: ERMatchingModel,
    matching_f,
    candidate_f,
    blocker_top_k: int = 10,
    threshold: float = 0.45,
    chunk_size: int = 250000
):
    """
    Executes streaming inference for a single country partition.
    Loads S2 and S3 for that country, builds inverted index, then streams S1 queries.
    """
    print(f"\n=======================================================")
    print(f"[*] Processing Country: {country}")
    print(f"=======================================================")

    blocker = InvertedIndexBlocker(max_postings_per_key=4000, top_k=blocker_top_k)

    # 1. Index S2 records for this country
    print(f"  -> Ingesting {country} records from Source 2...")
    s2_count = 0
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=chunk_size, dtype=str):
        c_chunk = chunk[chunk["country"] == country]
        if len(c_chunk) > 0:
            records = list(zip(c_chunk["entity_id"], c_chunk["business_name"], c_chunk["business_address"]))
            blocker.add_records(records)
            s2_count += len(c_chunk)
    print(f"     Indexed {s2_count} records from Source 2.")

    # 2. Index S3 records for this country
    print(f"  -> Ingesting {country} records from Source 3...")
    s3_count = 0
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=chunk_size, dtype=str):
        c_chunk = chunk[chunk["country"] == country]
        if len(c_chunk) > 0:
            records = list(zip(c_chunk["entity_id"], c_chunk["business_name"], c_chunk["business_address"]))
            blocker.add_records(records)
            s3_count += len(c_chunk)
    print(f"     Indexed {s3_count} records from Source 3.")
    print(f"     Total {country} index vocabulary keys: {len(blocker.index)}")

    # 3. Stream S1 records for this country, generate candidates, extract features, and score
    print(f"  -> Scoring {country} Source 1 entities in batched mode...")
    s1_processed = 0
    t0 = time.time()
    batch_size = 10000

    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=chunk_size, dtype=str):
        c_chunk = chunk[chunk["country"] == country]
        if len(c_chunk) == 0:
            continue

        records = list(zip(c_chunk["entity_id"], c_chunk["business_name"], c_chunk["business_address"]))
        for b_start in range(0, len(records), batch_size):
            b_records = records[b_start:b_start + batch_size]

            batch_features = []
            batch_pairs = []  # (s1_id, candidate_id)
            entity_candidates: Dict[str, List[str]] = {}

            for s1_id, name, addr in b_records:
                s1_stem, _ = extract_brand_stem(name)
                s1_clean_addr, nums, _ = extract_address_components(addr)
                s1_clean_nums = {n.lstrip("0") for n in nums if n.lstrip("0")}

                candidates = blocker.query_candidates(s1_stem, s1_clean_addr, s1_clean_nums)
                cand_ids = [c[0] for c in candidates]
                entity_candidates[s1_id] = cand_ids

                for cid, rank, h_score in candidates:
                    c_stem, c_addr, c_nums = blocker.entity_registry[cid]
                    feat = extract_pairwise_features(
                        s1_stem, s1_clean_addr, s1_clean_nums,
                        c_stem, c_addr, c_nums,
                        cid, rank, h_score
                    )
                    batch_features.append(feat)
                    batch_pairs.append((s1_id, cid))

            # Batch predict with LightGBM in fast C++
            entity_matches = defaultdict(list)
            if batch_features:
                probs = model.predict_proba(np.array(batch_features, dtype=np.float32))
                for (s1_id, cid), prob in zip(batch_pairs, probs):
                    if prob >= threshold:
                        entity_matches[s1_id].append(cid)

            # Stream write outputs for this batch
            for s1_id, _, _ in b_records:
                # Write candidate pairs
                cand_str = ",".join(entity_candidates.get(s1_id, []))
                candidate_f.write(f"{s1_id}\t{cand_str}\n")

                # Write matching results
                matched_str = ",".join(entity_matches.get(s1_id, []))
                matching_f.write(f"{s1_id}\t{matched_str}\n")

            s1_processed += len(b_records)
            if s1_processed % 50000 < batch_size or s1_processed == len(c_chunk):
                elapsed = time.time() - t0
                speed = s1_processed / elapsed if elapsed > 0 else 0
                print(f"     Processed {s1_processed:,} {country} entities ({speed:.1f} entities/sec)...")

    # Clear memory
    blocker.clear()
    print(f"  [+] Completed {country}: {s1_processed:,} S1 entities processed in {time.time()-t0:.1f}s.")


def train_baseline_model(
    train_dir: str,
    output_model_path: str,
    sample_entities: int = 15000
) -> Tuple[ERMatchingModel, float]:
    """
    Trains and calibrates the LightGBM ER matching model on a representative training sample.
    """
    print(f"\n[*] Training and Calibrating ER Matching Model on {sample_entities} entities...")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    # Read S1 sample
    s1_sample = pd.read_csv(s1_path, sep="\t", nrows=sample_entities, dtype=str)
    s1_ids = set(s1_sample["entity_id"])

    # Read ground truth for S1 sample
    gt_map = {}
    target_s23 = set()
    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=250000, dtype=str):
        matched_chunk = chunk[chunk["source1_entity_id"].isin(s1_ids)]
        for _, row in matched_chunk.iterrows():
            s1_id = row["source1_entity_id"]
            mids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip() and m.strip() != "nan"]
            gt_map[s1_id] = set(mids)
            for m in mids:
                target_s23.add(m)

    print(f"  Loaded GT for {len(gt_map)} S1 entities. Target match IDs: {len(target_s23)}")

    # Extract target S2 and S3 rows
    target_s2 = {m for m in target_s23 if m.startswith("S2-")}
    target_s3 = {m for m in target_s23 if m.startswith("S3-")}

    s2_rows = []
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=250000, dtype=str):
        found = chunk[chunk["entity_id"].isin(target_s2)]
        if len(found) > 0:
            s2_rows.append(found)
            target_s2 -= set(found["entity_id"])
            if not target_s2:
                break
    s2_df = pd.concat(s2_rows) if s2_rows else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    s3_rows = []
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=250000, dtype=str):
        found = chunk[chunk["entity_id"].isin(target_s3)]
        if len(found) > 0:
            s3_rows.append(found)
            target_s3 -= set(found["entity_id"])
            if not target_s3:
                break
    s3_df = pd.concat(s3_rows) if s3_rows else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    # Build blocker and index target records + distractors
    blocker = InvertedIndexBlocker(max_postings_per_key=4000, top_k=10)
    records = list(zip(s2_df["entity_id"], s2_df["business_name"], s2_df["business_address"])) + \
              list(zip(s3_df["entity_id"], s3_df["business_name"], s3_df["business_address"]))
    blocker.add_records(records)

    # Preprocess S1 records
    s1_records = []
    for _, row in s1_sample.iterrows():
        eid = row["entity_id"]
        stem, _ = extract_brand_stem(row["business_name"])
        c_addr, nums, _ = extract_address_components(row["business_address"])
        c_nums = {n.lstrip("0") for n in nums if n.lstrip("0")}
        s1_records.append((eid, stem, c_addr, c_nums))

    # Split train/val
    np.random.seed(42)
    s1_id_list = [r[0] for r in s1_records]
    n_train = int(len(s1_id_list) * 0.75)
    train_ids = set(s1_id_list[:n_train])
    val_ids = set(s1_id_list[n_train:])

    X_train, y_train = [], []
    X_val, y_val, val_pairs = [], [], []

    for s1_id, s1_stem, s1_addr, s1_nums in s1_records:
        true_set = gt_map.get(s1_id, set())
        candidates = blocker.query_candidates(s1_stem, s1_addr, s1_nums)

        for cid, rank, h_score in candidates:
            c_stem, c_addr, c_nums = blocker.entity_registry[cid]
            feat = extract_pairwise_features(
                s1_stem, s1_addr, s1_nums,
                c_stem, c_addr, c_nums,
                cid, rank, h_score
            )
            label = 1 if cid in true_set else 0

            if s1_id in train_ids:
                X_train.append(feat)
                y_train.append(label)
            else:
                X_val.append(feat)
                y_val.append(label)
                val_pairs.append((s1_id, cid))

    print(f"  Feature dataset: Train={len(X_train)} (pos={sum(y_train)}), Val={len(X_val)} (pos={sum(y_val)})")

    model = ERMatchingModel()
    model.train(np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val))

    val_probs = model.predict_proba(np.array(X_val))
    best_thresh = model.optimize_threshold(val_pairs, val_probs, list(val_ids), gt_map)
    print(f"  [+] Calibrated optimal decision threshold: {best_thresh:.2f}")

    # Evaluate validation score
    val_preds = defaultdict(set)
    for (s1_id, cid), prob in zip(val_pairs, val_probs):
        if prob >= best_thresh:
            val_preds[s1_id].add(cid)
    val_metrics = evaluate_macro_f05(val_preds, {s1_id: gt_map.get(s1_id, set()) for s1_id in val_ids})
    print(f"  [+] Validation Macro F_0.5 Score: {val_metrics['macro_f05']:.4f}")
    print(f"      Precision: {val_metrics['avg_precision']:.4f}, Recall: {val_metrics['avg_recall']:.4f}, Singleton Acc: {val_metrics['singleton_accuracy']:.4f}")

    model.save(output_model_path)
    print(f"  [+] Saved calibrated model to {output_model_path}")
    return model, best_thresh


def run_full_test_pipeline(
    test_dir: str,
    output_dir: str,
    model_path: str,
    threshold: float = 0.45,
    top_k: int = 10
):
    """
    Runs end-to-end inference over the complete test set, generating matching_results.tsv and candidate_pairs.tsv.
    """
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    model = ERMatchingModel(optimal_threshold=threshold)
    model.load(model_path)

    # Check if files already exist to support resumable execution
    existing_s1 = set()
    file_mode = "w"
    if os.path.exists(matching_path) and os.path.getsize(matching_path) > 0:
        with open(matching_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.split("\t", 1)
                if parts and parts[0].strip():
                    existing_s1.add(parts[0].strip())
        if existing_s1:
            print(f"  [i] Found {len(existing_s1):,} existing S1 entities already processed in {matching_path}. Resuming...")
            file_mode = "a"

    with open(matching_path, file_mode, encoding="utf-8") as matching_f, \
         open(candidate_path, file_mode, encoding="utf-8") as candidate_f:

        if file_mode == "w":
            matching_f.write("source1_entity_id\tmatched_entity_ids\n")
            candidate_f.write("source1_entity_id\tcandidate_entity_ids\n")

        countries = ["France", "US", "India"]

        for country in countries:
            # Check if this country was already fully processed
            # Check a sample S1 entity of this country
            sample_country_chunk = pd.read_csv(s1_path, sep="\t", nrows=1000, dtype=str)
            country_sample_ids = set(sample_country_chunk[sample_country_chunk["country"] == country]["entity_id"])
            if country_sample_ids and country_sample_ids.issubset(existing_s1):
                # Verify total count
                matching_count_in_existing = len(country_sample_ids & existing_s1)
                if matching_count_in_existing == len(country_sample_ids):
                    print(f"  [+] Country {country} already appears in existing output. Skipping.")
                    continue

            run_country_inference(
                country=country,
                s1_path=s1_path,
                s2_path=s2_path,
                s3_path=s3_path,
                model=model,
                matching_f=matching_f,
                candidate_f=candidate_f,
                blocker_top_k=top_k,
                threshold=threshold
            )

    print(f"\n[SUCCESS] Generated output files:")
    print(f"  - Matching Results:  {matching_path}")
    print(f"  - Candidate Pairs:   {candidate_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["train", "infer", "all"], default="all", help="Pipeline execution mode")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to training dataset folder")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test dataset folder")
    parser.add_argument("--output-dir", default="output", help="Path to output folder for TSVs")
    parser.add_argument("--model-path", default="model_lgb.txt", help="Path to save/load trained model")
    parser.add_argument("--sample-entities", type=int, default=5000, help="Number of training entities for calibration")
    parser.add_argument("--threshold", type=float, default=0.45, help="Decision threshold for matching")
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        model, best_thresh = train_baseline_model(
            train_dir=args.train_dir,
            output_model_path=args.model_path,
            sample_entities=args.sample_entities
        )
        args.threshold = best_thresh

    if args.mode in ["infer", "all"]:
        run_full_test_pipeline(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            threshold=args.threshold
        )
