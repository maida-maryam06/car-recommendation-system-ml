"""
DS1 Pipeline – Car Evaluation Dataset
======================================
Dataset : elikplim/car-evaluation-data-set  (kagglehub)
Steps   :
  1. Load & Preprocessing  (Label Encoding + Standard Scaling)
  2. Unsupervised Learning (KMeans + Agglomerative) + Validation
  3. Feature Selection     (Mutual Information – SelectKBest)
  4. Imbalance Handling    (SMOTE + RandomOverSampler)
  5. Supervised Classifiers (LR, DT, RF, GradientBoosting, AdaBoost)
  6. Metrics & Visualisation – saves every chart to outputs/ds1/
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

# ── sklearn ────────────────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)

# ── imbalanced-learn ───────────────────────────────────────────────────────────
from imblearn.over_sampling import SMOTE, RandomOverSampler

# ── output dir ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "ds1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ["#4f8ef7", "#6fcf97", "#f2994a", "#eb5757", "#9b51e0"]


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
    path = kagglehub.dataset_download("elikplim/car-evaluation-data-set")

    csv_path = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                csv_path = os.path.join(root, f)
                break
        if csv_path:
            break

    cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    df = pd.read_csv(csv_path, names=cols)
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution:\n{df['class'].value_counts()}\n")

    # Label encode every column
    encoders = {}
    df_enc = df.copy()
    le = LabelEncoder()
    for col in cols:
        df_enc[col] = le.fit_transform(df[col])
        enc = LabelEncoder().fit(df[col])
        encoders[col] = enc

    X = df_enc.drop("class", axis=1)
    y = df_enc["class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Class distribution chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))
    vc = df["class"].value_counts()
    bars = ax.bar(vc.index, vc.values, color=COLORS[:len(vc)])
    ax.set_title("DS1 · Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class"); ax.set_ylabel("Count")
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5,
                str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "01_class_distribution.png")

    return df, df_enc, encoders, X, y, X_scaled, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 2. UNSUPERVISED LEARNING
# ══════════════════════════════════════════════════════════════════════════════
def unsupervised(X_scaled):
    print("\n── Step 2 · Unsupervised Learning ───────────────────────────────")

    results = {}

    # ── KMeans ──────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)
    km_db  = davies_bouldin_score(X_scaled, km_labels)
    print(f"  KMeans      → Silhouette: {km_sil:.4f}  Davies-Bouldin: {km_db:.4f}")
    results["kmeans"] = {"silhouette": km_sil, "davies_bouldin": km_db}

    # ── Agglomerative ────────────────────────────────────────────────────────
    agg = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = silhouette_score(X_scaled, agg_labels)
    agg_db  = davies_bouldin_score(X_scaled, agg_labels)
    print(f"  Agglomerative → Silhouette: {agg_sil:.4f}  Davies-Bouldin: {agg_db:.4f}")
    results["agglomerative"] = {"silhouette": agg_sil, "davies_bouldin": agg_db}

    # ── Elbow curve ───────────────────────────────────────────────────────────
    inertias = []
    K = range(2, 10)
    for k in K:
        inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K), inertias, "o-", color="#4f8ef7", linewidth=2, markersize=7)
    ax.fill_between(list(K), inertias, alpha=0.07, color="#4f8ef7")
    ax.set_title("KMeans · Elbow Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Clusters (k)"); ax.set_ylabel("Inertia")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "02_kmeans_elbow.png")

    # ── Cluster validation comparison bar ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    algos = ["KMeans", "Agglomerative"]
    sils  = [km_sil, agg_sil]
    dbs   = [km_db,  agg_db]

    axes[0].bar(algos, sils, color=["#4f8ef7", "#6fcf97"])
    axes[0].set_title("Silhouette Score (higher = better)")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(sils):
        axes[0].text(i, v + 0.01, f"{v:.4f}", ha="center")

    axes[1].bar(algos, dbs, color=["#4f8ef7", "#6fcf97"])
    axes[1].set_title("Davies-Bouldin Index (lower = better)")
    for i, v in enumerate(dbs):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center")

    fig.suptitle("Clustering Validation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_cluster_validation.png")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def feature_selection(X, y):
    print("\n── Step 3 · Feature Selection ───────────────────────────────────")

    selector = SelectKBest(mutual_info_classif, k=4)
    X_sel = selector.fit_transform(X, y)
    selected = list(X.columns[selector.get_support()])
    scores   = selector.scores_
    print(f"  Selected features: {selected}")

    # Feature importance chart
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_idx = np.argsort(scores)[::-1]
    sorted_feats  = [X.columns[i] for i in sorted_idx]
    sorted_scores = [scores[i] for i in sorted_idx]
    colors = ["#4f8ef7" if X.columns[i] in selected else "#444c6a" for i in sorted_idx]
    bars = ax.bar(sorted_feats, sorted_scores, color=colors)
    ax.set_title("Feature Selection · Mutual Information Scores", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature"); ax.set_ylabel("MI Score")
    for b, v in zip(bars, sorted_scores):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4f8ef7", label="Selected"),
                        Patch(color="#444c6a", label="Dropped")])
    fig.tight_layout()
    save_fig(fig, "04_feature_selection.png")

    return selector, X_sel, selected


# ══════════════════════════════════════════════════════════════════════════════
# 4. IMBALANCE HANDLING
# ══════════════════════════════════════════════════════════════════════════════
def handle_imbalance(X_sel, y):
    print("\n── Step 4 · Imbalance Handling ──────────────────────────────────")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sel, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_sm, y_sm = smote.fit_resample(X_tr, y_tr)

    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X_tr, y_tr)

    print(f"  Original  train: {dict(pd.Series(y_tr).value_counts())}")
    print(f"  After SMOTE    : {dict(pd.Series(y_sm).value_counts())}")
    print(f"  After ROS      : {dict(pd.Series(y_ros).value_counts())}")

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    titles  = ["Original Train", "After SMOTE", "After ROS"]
    series  = [pd.Series(y_tr), pd.Series(y_sm), pd.Series(y_ros)]

    for ax, title, s in zip(axes, titles, series):
        vc = s.value_counts()
        ax.bar(vc.index.astype(str), vc.values, color=COLORS[:len(vc)])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Class"); ax.set_ylabel("Count")

    fig.suptitle("Class Distribution Before / After Balancing", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_imbalance_handling.png")

    return X_sm, y_sm, X_te, y_te


# ══════════════════════════════════════════════════════════════════════════════
# 5. SUPERVISED CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════
def supervised_classifiers(X_train, y_train, X_test, y_test):
    print("\n── Step 5 · Supervised Classifiers ──────────────────────────────")

    classifiers = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":        DecisionTreeClassifier(random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost":             AdaBoostClassifier(n_estimators=100, random_state=42),
    }

    all_results = {}
    best_clf  = None
    best_name = ""
    best_acc  = 0.0

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cr   = classification_report(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        print(f"\n  {name}")
        print(f"    Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        print(f"    Classification Report:\n{cr}")

        # ── Confusion matrix image ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    linewidths=.5, linecolor="white")
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.tight_layout()
        fname = "cm_" + name.lower().replace(" ", "_") + ".png"
        save_fig(fig, fname)

        all_results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "clf": clf, "cm": cm, "report": cr,
        }

        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_clf  = clf

    # ── Accuracy comparison bar ────────────────────────────────────────────
    names = list(all_results.keys())
    accs  = [all_results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, accs, color=COLORS)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy"); ax.set_title("Classifier Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.axhline(y=max(accs), color="white", linestyle="--", alpha=0.4)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{a:.2%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticklabels(names, rotation=18, ha="right")
    fig.tight_layout()
    save_fig(fig, "06_accuracy_comparison.png")

    # ── Full metrics grouped bar ───────────────────────────────────────────
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (metric, color) in enumerate(zip(metrics, COLORS)):
        vals = [all_results[n][metric] for n in names]
        ax.bar(x + i * width, vals, width, label=metric.capitalize(), color=color)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score"); ax.set_title("All Metrics · All Classifiers", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "07_all_metrics.png")

    print(f"\n  Best Classifier: {best_name} (Accuracy: {best_acc:.4f})")
    return all_results, best_clf, best_name


# ══════════════════════════════════════════════════════════════════════════════
# MAIN – run entire DS1 pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run(return_model=False):
    print("=" * 60)
    print("  DS1 PIPELINE · Car Evaluation Dataset")
    print("=" * 60)

    df, df_enc, encoders, X, y, X_scaled, scaler = load_and_preprocess()
    cluster_results = unsupervised(X_scaled)
    selector, X_sel, selected_features = feature_selection(X, y)
    X_train, y_train, X_test, y_test = handle_imbalance(X_sel, y)
    clf_results, best_clf, best_name = supervised_classifiers(X_train, y_train, X_test, y_test)

    print("\n✅  DS1 pipeline complete. Charts saved to outputs/ds1/")

    if return_model:
        return {
            "encoders": encoders, "scaler": scaler,
            "selector": selector, "best_clf": best_clf,
            "best_name": best_name, "selected_features": selected_features,
        }


if __name__ == "__main__":
    run()