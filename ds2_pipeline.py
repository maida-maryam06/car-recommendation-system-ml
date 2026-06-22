"""
DS2 Pipeline – Craigslist Cars & Trucks Dataset
=================================================
Dataset : austinreese/craigslist-carstrucks-data  (kagglehub)
Steps   :
  1. Load & Preprocess  (50K row sample, price bucketing, encoding)
  2. Feature Extraction (PCA + SelectKBest Mutual Information)
  3. Classification     (Random Forest + Gradient Boosting)
  4. Metrics & Visualisation – saves every chart to outputs/ds2/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

# ── output dir ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "ds2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ["#4f8ef7", "#6fcf97", "#f2994a", "#eb5757", "#9b51e0"]
NROWS  = 50_000   # sample size (full file is ~426K – avoids RAM overload)


# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess():
    print("\n── Step 1 · Load & Preprocess ────────────────────────────────────")
    import kagglehub
    path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")

    csv_path = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                csv_path = os.path.join(root, f)
                break
        if csv_path:
            break

    df = pd.read_csv(csv_path, nrows=NROWS, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Keep useful columns
    use_cols = ["price", "year", "odometer", "condition", "fuel",
                "title_status", "transmission", "drive", "type", "paint_color"]
    df = df[use_cols].copy()
    df.dropna(inplace=True)
    df = df[df["price"].between(500, 100_000)]
    df = df[df["odometer"] < 500_000]
    df = df[df["year"].between(1980, 2024)]
    print(f"  Cleaned shape: {df.shape}")

    # Price bucket label
    df["price_cat"] = pd.cut(
        df["price"],
        bins=[0, 5_000, 15_000, 30_000, 100_000],
        labels=["Budget", "Mid", "Premium", "Luxury"],
    )
    df.dropna(subset=["price_cat"], inplace=True)

    # ── Distribution chart ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    vc = df["price_cat"].value_counts()
    axes[0].bar(vc.index, vc.values, color=COLORS[:len(vc)])
    axes[0].set_title("DS2 · Price Category Distribution", fontweight="bold")
    axes[0].set_xlabel("Category"); axes[0].set_ylabel("Count")

    axes[1].hist(df["price"], bins=50, color="#4f8ef7", edgecolor="none", alpha=0.85)
    axes[1].set_title("DS2 · Price Histogram (filtered)", fontweight="bold")
    axes[1].set_xlabel("Price ($)"); axes[1].set_ylabel("Count")

    fig.tight_layout()
    save_fig(fig, "01_data_distribution.png")

    # Encode categoricals
    cat_cols = ["condition","fuel","title_status","transmission","drive","type","paint_color"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(["price", "price_cat"], axis=1)
    y = LabelEncoder().fit_transform(df["price_cat"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Feature matrix: {X.shape}  Label distribution: {dict(pd.Series(y).value_counts())}")
    return df, X, y, X_scaled, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 2a. FEATURE EXTRACTION – PCA
# ══════════════════════════════════════════════════════════════════════════════
def extract_pca(X_scaled, n=5):
    print(f"\n── Feature Extraction 1 · PCA (n={n}) ──────────────────────────")

    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_
    cum_ev = np.cumsum(ev)
    print(f"  Explained variance per component: {np.round(ev, 4)}")
    print(f"  Cumulative:                       {np.round(cum_ev, 4)}")

    # ── Explained variance chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    comps = [f"PC{i+1}" for i in range(n)]
    bars = ax.bar(comps, ev * 100, color="#4f8ef7", label="Individual")
    ax.plot(comps, cum_ev * 100, "o-", color="#f2994a", linewidth=2, label="Cumulative")
    ax.set_title("PCA · Explained Variance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component"); ax.set_ylabel("Explained Variance (%)")
    for b, v in zip(bars, ev * 100):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "02_pca_explained_variance.png")

    # ── PCA scatter (PC1 vs PC2) ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    y_dummy = np.zeros(X_pca.shape[0])  # no label needed here
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=y_dummy, cmap="Blues", alpha=0.3, s=5)
    ax.set_title("PCA · PC1 vs PC2 Scatter", fontsize=13, fontweight="bold")
    ax.set_xlabel("Principal Component 1"); ax.set_ylabel("Principal Component 2")
    fig.tight_layout()
    save_fig(fig, "03_pca_scatter.png")

    return pca, X_pca


# ══════════════════════════════════════════════════════════════════════════════
# 2b. FEATURE EXTRACTION – SelectKBest (Mutual Info)
# ══════════════════════════════════════════════════════════════════════════════
def extract_selectkbest(X, y, k=5):
    print(f"\n── Feature Extraction 2 · SelectKBest MI (k={k}) ───────────────")

    skb = SelectKBest(mutual_info_classif, k=k)
    X_skb = skb.fit_transform(X, y)
    selected = list(X.columns[skb.get_support()])
    scores   = skb.scores_
    print(f"  Selected features: {selected}")

    # MI score chart
    sorted_idx    = np.argsort(scores)[::-1]
    sorted_feats  = [X.columns[i] for i in sorted_idx]
    sorted_scores = [scores[i] for i in sorted_idx]
    colors = ["#4f8ef7" if X.columns[i] in selected else "#444c6a" for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(sorted_feats, sorted_scores, color=colors)
    ax.set_title("DS2 · Feature Extraction – Mutual Information Scores",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature"); ax.set_ylabel("MI Score")
    ax.set_xticklabels(sorted_feats, rotation=20, ha="right")
    for b, v in zip(bars, sorted_scores):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4f8ef7", label="Selected"),
                        Patch(color="#444c6a", label="Dropped")])
    fig.tight_layout()
    save_fig(fig, "04_selectkbest_mi.png")

    return skb, X_skb, selected


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFICATION (on PCA features)
# ══════════════════════════════════════════════════════════════════════════════
def classify(X_pca, y):
    print("\n── Step 3 · Classification (on PCA features) ────────────────────")

    X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    classifiers = {
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    all_results = {}
    for name, clf in classifiers.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_te, y_pred)
        cr   = classification_report(y_te, y_pred, zero_division=0)

        print(f"\n  {name}")
        print(f"    Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        print(f"    Report:\n{cr}")

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=.5, linecolor="white")
        ax.set_title(f"DS2 · {name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.tight_layout()
        fname = "cm_" + name.lower().replace(" ", "_") + ".png"
        save_fig(fig, fname)

        all_results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Comparison chart
    names   = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (metric, color) in enumerate(zip(metrics, COLORS)):
        vals = [all_results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("DS2 · Classifier Performance After Feature Extraction",
                 fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "05_classifier_comparison.png")

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run():
    print("=" * 60)
    print("  DS2 PIPELINE · Craigslist Cars & Trucks")
    print("=" * 60)

    df, X, y, X_scaled, scaler = load_and_preprocess()
    pca, X_pca = extract_pca(X_scaled, n=5)
    skb, X_skb, selected = extract_selectkbest(X, y, k=5)
    results = classify(X_pca, y)

    print("\n✅  DS2 pipeline complete. Charts saved to outputs/ds2/")
    return results


if __name__ == "__main__":
    run()