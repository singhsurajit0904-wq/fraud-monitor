from __future__ import annotations

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Card Fraud Monitor", page_icon="💳", layout="wide")
st.title("Credit Card Fraud Detection")
st.caption("A starter decision-support tool: estimate fraud risk, choose an alert threshold, and inspect model signals.")
MODEL_PATH = Path("fraud_model.joblib")


def read_table(upload) -> pd.DataFrame:
    # The provided pasted dataset is tab-separated; delimiter sniffing also accepts CSV.
    return pd.read_csv(upload, sep=None, engine="python")


def validate_training(df: pd.DataFrame):
    if "Class" not in df.columns:
        raise ValueError("Training data needs a 'Class' column (0 = legitimate, 1 = fraud).")
    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("'Class' must contain only 0 and 1.")
    features = [c for c in df.columns if c != "Class"]
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["Class"], errors="coerce")
    keep = y.notna() & X.notna().all(axis=1)
    X, y = X.loc[keep], y.loc[keep].astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Need at least two usable rows from each class.")
    return X, y


def split_indices(X, y):
    # Preserve transaction order when Time is present and non-decreasing.
    if "Time" in X and X["Time"].is_monotonic_increasing:
        cut = int(len(X) * 0.8)
        return np.arange(cut), np.arange(cut, len(X)), "chronological 80/20 (Time order)"
    from sklearn.model_selection import train_test_split
    a, b = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    return a, b, "stratified random 80/20 (Time was not monotonic)"


tab_train, tab_score = st.tabs(["Train / evaluate", "Score transactions"])
with tab_train:
    st.subheader("1. Upload the labeled training data")
    st.write("Expected target column: `Class` (0 = legitimate, 1 = fraud). The supplied sample uses `Time`, `V1`–`V28`, `Amount`, and `Class`.")
    train_file = st.file_uploader("Labeled CSV or TSV", type=["csv", "tsv", "txt"], key="train")
    left, right = st.columns(2)
    with left:
        fp_cost = st.number_input("Cost of one false alert", min_value=0.0, value=1.0, step=1.0)
    with right:
        fn_cost = st.number_input("Cost of one missed fraud", min_value=0.0, value=10.0, step=1.0)
    if train_file is not None:
        try:
            data = read_table(train_file)
            st.write(f"Loaded {len(data):,} rows and {len(data.columns)} columns.")
            st.dataframe(data.head(), use_container_width=True)
            if st.button("Train baseline and save model", type="primary"):
                X, y = validate_training(data)
                tr, va, split_note = split_indices(X, y)
                model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
                model.fit(X.iloc[tr], y.iloc[tr])
                probs = model.predict_proba(X.iloc[va])[:, 1]
                yv = y.iloc[va].to_numpy()
                thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 99), probs])
                best_t, best_cost = 0.5, float("inf")
                for threshold in thresholds:
                    pred = probs >= threshold
                    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
                    cost = fp_cost * fp + fn_cost * fn
                    if cost < best_cost:
                        best_t, best_cost = float(threshold), float(cost)
                pred = probs >= best_t
                tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
                model_bundle = {"model": model, "features": list(X.columns), "threshold": best_t}
                joblib.dump(model_bundle, MODEL_PATH)
                st.success(f"Model saved to {MODEL_PATH.resolve()}")
                st.info(f"Validation split: {split_note}. Threshold selected to minimize the costs you entered.")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Average precision (PR-AUC)", f"{average_precision_score(yv, probs):.4f}")
                try:
                    m2.metric("ROC-AUC", f"{roc_auc_score(yv, probs):.4f}")
                except ValueError:
                    m2.metric("ROC-AUC", "Unavailable")
                m3.metric("Precision at selected threshold", f"{precision_score(yv, pred, zero_division=0):.4f}")
                m4.metric("Recall at selected threshold", f"{recall_score(yv, pred, zero_division=0):.4f}")
                st.write(f"Threshold: **{best_t:.4f}** · Validation cost: **{best_cost:,.2f}** · TP {tp:,}, FP {fp:,}, FN {fn:,}, TN {tn:,}")
                coef = model.named_steps["logisticregression"].coef_[0]
                importance = pd.DataFrame({"Feature": X.columns, "Absolute standardized coefficient": np.abs(coef), "Direction": np.where(coef >= 0, "higher score", "lower score")}).sort_values("Absolute standardized coefficient", ascending=False).head(12)
                st.subheader("Model signals (coefficient-based, not causal explanations)")
                st.bar_chart(importance.set_index("Feature")["Absolute standardized coefficient"])
                st.dataframe(importance, hide_index=True, use_container_width=True)
                st.caption("This baseline does not detect novel fraud patterns by itself. Validation metrics are from this dataset split and may not reflect live performance.")
        except Exception as exc:
            st.error(str(exc))

with tab_score:
    st.subheader("2. Score new transactions")
    st.write("Upload rows containing the same feature columns used during training, without `Class`. This is transaction scoring, not a forecast of future time periods.")
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        score_file = st.file_uploader("Unlabeled transactions CSV or TSV", type=["csv", "tsv", "txt"], key="score")
        if score_file is not None:
            try:
                scoring = read_table(score_file)
                missing = [c for c in bundle["features"] if c not in scoring.columns]
                if missing:
                    st.error("Missing model input columns: " + ", ".join(missing))
                else:
                    Xnew = scoring[bundle["features"]].apply(pd.to_numeric, errors="coerce")
                    if Xnew.isna().any().any():
                        st.error("Some model inputs are missing or non-numeric. Fix those rows before scoring.")
                    elif st.button("Score uploaded transactions", type="primary"):
                        risk = bundle["model"].predict_proba(Xnew)[:, 1]
                        result = scoring.copy()
                        result["fraud_probability"] = risk
                        result["alert"] = (risk >= bundle["threshold"]).astype(int)
                        st.dataframe(result.head(100), use_container_width=True)
                        st.write(f"Alerts: **{int(result['alert'].sum()):,}** of {len(result):,} transactions · threshold {bundle['threshold']:.4f}")
                        fig, ax = plt.subplots()
                        ax.hist(risk, bins=30, color="#4c78a8")
                        ax.axvline(bundle["threshold"], color="red", linestyle="--", label="Alert threshold")
                        ax.set(xlabel="Estimated fraud probability", ylabel="Transactions", title="Transaction risk scores")
                        ax.legend()
                        st.pyplot(fig)
                        st.download_button("Download scored CSV", result.to_csv(index=False).encode("utf-8"), "scored_transactions.csv", "text/csv")
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("Train and save a model in the first tab before scoring transactions.")

st.divider()
st.caption("Operational use requires organizer-approved data handling, cost assumptions, and validation. The supplied challenge rules, official metric, and submission template were not provided with the pasted dataset.")
