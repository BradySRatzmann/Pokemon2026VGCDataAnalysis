import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from collections import Counter

def build_sixth_mon_dataset(team_df: pd.DataFrame):
    """
    Returns:
      X_tokens: list[list[str]] where each inner list is the 5 known mons
      y: list[str] the held-out mon to predict
    """
    cols = [f"pokemon_{i}" for i in range(1, 7)]
    X_tokens: list[list[str]] = []
    y: list[str] = []

    for _, row in team_df.iterrows():
        mons = []
        for c in cols:
            v = row.get(c)
            if pd.notna(v) and v:
                mons.append(str(v).strip().lower())

        # only train on proper teams
        mons = sorted(set(mons))
        if len(mons) != 6:
            continue

        # make 6 training examples per team
        for i in range(6):
            target = mons[i]
            known = [m for j, m in enumerate(mons) if j != i]
            X_tokens.append(known)
            y.append(target)

    return X_tokens, y

def train_sixth_mon_model(
    team_df: pd.DataFrame,
    random_state: int = 0,
    min_label_count: int = 2,   # try 5 or 10 for cleaner, more stable classes
):
    X_tokens, y = build_sixth_mon_dataset(team_df)

    # --- Filter rare labels (prevents stratify errors + improves stability) ---
    counts = Counter(y)
    keep_mask = [counts[label] >= min_label_count for label in y]
    X_tokens = [x for x, keep in zip(X_tokens, keep_mask) if keep]
    y = [label for label, keep in zip(y, keep_mask) if keep]

    if len(y) == 0 or len(set(y)) < 2:
        raise ValueError(
            "Not enough training data after filtering rare labels. "
            "Lower min_label_count or increase dataset size."
        )

    # --- Vectorize 5-mon inputs as multi-hot features ---
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(X_tokens)

    # --- Split (only stratify if every remaining class has >=2) ---
    remaining_counts = Counter(y)
    stratify_arg = y if min(remaining_counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=stratify_arg,
    )

    # --- Train multiclass model (new sklearn: no multi_class kwarg) ---
    clf = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    # --- Evaluate with Top-K accuracy (aligned to full class list) ---
    probs = clf.predict_proba(X_test)
    labels_all = clf.classes_

    top1 = top_k_accuracy_score(y_test, probs, k=1, labels=labels_all)
    top3 = top_k_accuracy_score(y_test, probs, k=3, labels=labels_all)
    top5 = top_k_accuracy_score(y_test, probs, k=5, labels=labels_all)

    metrics = {
        "top1": float(top1),
        "top3": float(top3),
        "top5": float(top5),
        "min_label_count": int(min_label_count),
        "num_classes": int(len(labels_all)),
        "num_examples": int(len(y)),
    }

    return clf, mlb, metrics


def recommend_sixth_mon(
    clf: LogisticRegression,
    mlb: MultiLabelBinarizer,
    known_mons: list[str],
    top_n: int = 10,
):
    known = sorted(set([m.strip().lower() for m in known_mons if m]))
    X = mlb.transform([known])

    probs = clf.predict_proba(X)[0]
    classes = clf.classes_

    # Don't recommend something already on the team
    scores = [(classes[i], float(probs[i])) for i in range(len(classes)) if classes[i] not in known]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]

