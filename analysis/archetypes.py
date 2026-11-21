import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MultiLabelBinarizer, Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# Use of Scikit-Learn for Advanced Analysis of team archetypes
# -----------------------------------------------------------------------------
def build_team_lists_from_team_df(team_df: pd.DataFrame) -> tuple[list[int], list[list[str]]]:
    cols = [f"pokemon_{i}" for i in range(1, 7)]
    team_ids = []
    teams = []

    for _, row in team_df.iterrows():
        team_id = int(row["team_id"])
        mons = []
        for c in cols:
            m = row.get(c)
            if pd.notna(m) and m:
                mons.append(str(m).strip().lower())

        mons = sorted(set(mons))  # presence-only
        team_ids.append(team_id)
        teams.append(mons)

    return team_ids, teams

def run_archetype_clustering(
    team_df: pd.DataFrame,
    output_root: str = "outputs",
    k_min: int = 4,
    k_max: int = 20,
    svd_components: int = 50,
    random_state: int = 0,
):
    team_ids, teams = build_team_lists_from_team_df(team_df)

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(teams)      # shape: (n_teams, n_pokemon)
    vocab = mlb.classes_

    # Dimensionality reduction (cap components safely)
    n_comp = min(svd_components, max(2, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    X_red = svd.fit_transform(X)

    # Normalization helps KMeans in practice
    X_red = Normalizer().fit_transform(X_red)


    # Choose K with silhouette (export all scores so you can justify in writeup)
    k_scores = []
    best = None

    # Determine how many teams are available
    n_samples = X_red.shape[0]

    if n_samples < 3:
        raise ValueError(
            "At least 3 teams are required for archetype clustering."
        )

    # Silhouette scoring requires k to be smaller than the number of teams.
    # If the dataset is small, lower the minimum cluster count to 2.
    if n_samples <= k_min:
        k_min = 2

    k_max = min(k_max, n_samples - 1)

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_red)

        # Silhouette requires at least 2 clusters and fewer clusters than teams.
        n_labels = len(set(labels))

        if 2 <= n_labels < n_samples:
            score = silhouette_score(X_red, labels)
        else:
            score = -1.0

        k_scores.append((k, float(score)))

        if best is None or score > best[1]:
            best = (k, score, km, labels)


    best_k, best_score, km, labels = best

    # Output folders
    out_dir = os.path.join(output_root, "ml", "team_archetypes_clustering")
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(k_scores, columns=["k", "silhouette"]).to_csv(
        os.path.join(out_dir, "k_silhouette.csv"), index=False
    )

    team_clusters = pd.DataFrame({"team_id": team_ids, "cluster": labels})
    team_clusters.to_csv(os.path.join(out_dir, "team_clusters.csv"), index=False)

    # Cluster summaries: identify the most distinctive Pokémon
    summaries = []
    X_arr = X  # numpy array already

    # Overall usage rate for each Pokémon across all teams
    overall_rates = X_arr.mean(axis=0)

    for c in range(best_k):
        idx = np.where(labels == c)[0]

        if len(idx) == 0:
            continue

        # Count Pokémon appearances within this cluster
        counts = X_arr[idx].sum(axis=0)

        # Usage rate within this cluster
        cluster_rates = counts / len(idx)

        # Calculate how distinctive each Pokémon is to this cluster
        # A higher lift means the Pokémon is more common here
        # than it is across the overall dataset.
        lift = np.divide(
            cluster_rates,
            overall_rates,
            out=np.zeros_like(cluster_rates, dtype=float),
            where=overall_rates > 0
        )

        # Combine cluster usage and distinctiveness into one score
        core_scores = cluster_rates * np.log2(
            np.maximum(lift, 1e-10)
        )

        # Prefer Pokémon that appear on at least 25% of cluster teams
        # and at least 2% of all teams.
        eligible = (
                (cluster_rates >= 0.25) &
                (overall_rates >= 0.02) &
                (lift > 1)
        )

        # If no Pokémon meet those thresholds, use any Pokémon
        # that is more common in this cluster than overall.
        if not eligible.any():
            eligible = lift > 1

        # Rank Pokémon by their core score
        ranked_idx = np.argsort(
            np.where(eligible, core_scores, -np.inf)
        )[::-1]

        core_idx = [
            i for i in ranked_idx
            if eligible[i]
        ][:2]

        core_pokemon = [vocab[i] for i in core_idx]

        # Generate a readable archetype name
        if core_pokemon:
            archetype_name = (
                    " - ".join(
                        pokemon.replace("-", " ").title()
                        for pokemon in core_pokemon
                    )
                    + " Core"
            )
        else:
            archetype_name = f"Archetype {c + 1}"

        # Keep the original top Pokémon summary
        top_idx = np.argsort(-counts)[:15]

        top_mons = [
            f"{vocab[i]}({int(counts[i])})"
            for i in top_idx
            if counts[i] > 0
        ]

        summaries.append({
            "cluster": int(c),
            "archetype_name": archetype_name,
            "core_pokemon": "; ".join(core_pokemon),
            "num_teams": int(len(idx)),
            "top_mons": "; ".join(top_mons),
        })

    # Sort clusters from most common to least common
    cluster_summaries = pd.DataFrame(summaries).sort_values(
        "num_teams",
        ascending=False
    ).reset_index(drop=True)

    cluster_summaries.to_csv(
        os.path.join(out_dir, "cluster_summaries.csv"),
        index=False
    )

    return best_k, float(best_score), cluster_summaries, team_clusters