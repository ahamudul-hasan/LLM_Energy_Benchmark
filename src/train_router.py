"""
Router Offline Training & Calibration Pipeline.

Performs offline model training and confidence threshold calibration per main.pdf Section 3.2:
1. Ingests 3,000 training prompts from dataset partitions across 6 datasets.
2. Extracts hybrid CPU feature vectors x in R^d (heuristics + sentence embeddings).
3. Evaluates Logistic Regression (C=1.0) vs Random Forest (100 estimators) backends using 80/20 train-val split and 5-fold cross-validation.
4. Performs grid-search calibration on validation split to select confidence thresholds (tau1, tau2) ensuring False Negative Rate (FNR) < 5%.
5. Serializes trained model pipeline and calibrated parameters.
"""

import os
import json
import time
import glob
import numpy as np
from typing import Dict, Any, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib

import sys
sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import load_jsonl
from feature_extractor import CPUFeatureExtractor


def load_training_corpus(training_dir: str = os.path.join("data", "training")) -> Tuple[List[str], np.ndarray]:
    """Loads training prompts and target tier ground-truth labels."""
    files = glob.glob(os.path.join(training_dir, "*_train.jsonl"))
    prompts = []
    labels = []

    for f in files:
        items = load_jsonl(f)
        for item in items:
            prompts.append(item["prompt"])
            labels.append(int(item["tier"]))

    if not prompts:
        raise FileNotFoundError(f"No training jsonl files found in {training_dir}")

    return prompts, np.array(labels, dtype=np.int32)


def calibrate_thresholds(
    probs: np.ndarray,
    y_val: np.ndarray,
    target_max_fnr: float = 0.05
) -> Tuple[float, float]:
    """
    Grid-search calibration for confidence escalation thresholds (tau1, tau2).
    Tau1: Tier 1 -> Tier 2 escalation
    Tau2: Tier 2 -> Tier 3 escalation
    Minimizes False Negative Rate (FNR) below target_max_fnr (5%).
    """
    best_tau1, best_tau2 = 0.5, 0.5
    min_fnr = 1.0

    # Grid search over probability thresholds [0.10, 0.80]
    grid = np.linspace(0.1, 0.8, 20)

    for t1 in grid:
        for t2 in grid:
            preds = []
            for i in range(len(probs)):
                p1, p2, p3 = probs[i][0], probs[i][1], probs[i][2]

                # Decision logic with confidence thresholds
                if p1 >= t1 and p1 >= p2 and p1 >= p3:
                    pred = 1
                elif (p1 + p2) >= t2 and p2 >= p3:
                    pred = 2
                else:
                    pred = 3
                preds.append(pred)

            preds = np.array(preds)

            # False negative: actual high complexity (tier 3) predicted as lower (tier 1 or 2)
            fn_mask = (y_val == 3) & (preds < 3)
            fn_count = np.sum(fn_mask)
            total_high = np.sum(y_val == 3)

            fnr = (fn_count / total_high) if total_high > 0 else 0.0

            if fnr <= target_max_fnr:
                return round(float(t1), 3), round(float(t2), 3)

            if fnr < min_fnr:
                min_fnr = fnr
                best_tau1, best_tau2 = t1, t2

    return round(float(best_tau1), 3), round(float(best_tau2), 3)


def train_and_calibrate():
    print("--- Starting Router Offline Training & Calibration ---")

    # 1. Load training dataset
    prompts, y = load_training_corpus()
    print(f"Loaded {len(prompts)} training prompts across tiers: {np.bincount(y)}")

    # 2. Extract CPU feature representations
    print("Extracting CPU feature vectors (heuristics + MiniLM embeddings)...")
    extractor = CPUFeatureExtractor(use_embeddings=True)  # Hybrid feature pipeline
    X = extractor.transform_batch(prompts)
    print(f"Feature matrix shape: {X.shape}")

    # 3. Train-Validation Split (80-20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Model Candidates
    clf_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # 5. 5-Fold Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_lr = cross_val_score(clf_lr, X_train, y_train, cv=cv, scoring='f1_macro')
    scores_rf = cross_val_score(clf_rf, X_train, y_train, cv=cv, scoring='f1_macro')

    print(f"5-Fold CV Macro-F1 | Logistic Regression: {scores_lr.mean():.4f} +/- {scores_lr.std():.4f}")
    print(f"5-Fold CV Macro-F1 | Random Forest:       {scores_rf.mean():.4f} +/- {scores_rf.std():.4f}")

    # Select best backend
    if scores_lr.mean() >= scores_rf.mean():
        selected_clf = clf_lr
        clf_name = "Logistic Regression"
    else:
        selected_clf = clf_rf
        clf_name = "Random Forest"

    print(f"Selected Backend: {clf_name}")
    selected_clf.fit(X_train, y_train)

    # 6. Predict probabilities on validation set for threshold calibration
    probs_val = selected_clf.predict_proba(X_val)

    # Note: ensure proba columns match classes [1, 2, 3]
    classes = list(selected_clf.classes_)
    probs_aligned = np.zeros((len(probs_val), 3))
    for idx, c in enumerate(classes):
        probs_aligned[:, c - 1] = probs_val[:, idx]

    # 7. Threshold Calibration (FNR < 5%)
    tau1, tau2 = calibrate_thresholds(probs_aligned, y_val, target_max_fnr=0.05)
    print(f"Calibrated Escalation Thresholds: tau1 = {tau1}, tau2 = {tau2}")

    # Save model checkpoint
    model_payload = {
        "classifier": selected_clf,
        "clf_name": clf_name,
        "classes": list(selected_clf.classes_),
        "tau1": tau1,
        "tau2": tau2,
        "use_embeddings": True
    }

    save_path = os.path.join("src", "router_model.joblib")
    joblib.dump(model_payload, save_path)
    print(f"Saved router model checkpoint to {save_path}")


if __name__ == "__main__":
    train_and_calibrate()
