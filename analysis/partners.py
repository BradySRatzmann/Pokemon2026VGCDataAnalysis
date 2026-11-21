import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

# Pair (Synergy) Analysis
# -----------------------------------------------------------------------------


# Compute pair usage: how many teams used both Pokémon in the pair for all unordered pairs
def compute_pair_usage(team_df: pd.DataFrame) -> pd.DataFrame:
    pair_counts: dict[tuple[str, str], int] = {}

    total_teams = team_df["team_id"].nunique()

    for _, row in team_df.iterrows():
        # Collect non-empty Pokémon names from pokemon_1..pokemon_6
        mons = []
        for i in range(1, 7):
            name = row.get(f"pokemon_{i}")
            if pd.notna(name) and name:
                mons.append(name)

        # Unique + sorted so pairs are stable and not double-counted per team
        mons = sorted(set(mons))
        n = len(mons)

        # Plain nested loops to generate unordered pairs (no itertools)
        for i in range(n):
            for j in range(i + 1, n):
                a = mons[i]
                b = mons[j]
                key = (a, b)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    if not pair_counts:
        return pd.DataFrame(columns=["teams_used", "usage_rate"])

    pairs = list(pair_counts.keys())
    counts = list(pair_counts.values())

    index = pd.MultiIndex.from_tuples(pairs, names=["pokemon_1", "pokemon_2"])
    pair_df = pd.DataFrame({"teams_used": counts}, index=index)

    pair_df["usage_rate"] = pair_df["teams_used"] / total_teams * 100.0
    pair_df = pair_df.sort_values("teams_used", ascending=False)

    return pair_df


# Builds a DataFrame matrix for pair usage heatmap
def build_pair_matrix(pair_df: pd.DataFrame, mons: list[str]) -> pd.DataFrame:
    mat = pd.DataFrame(0.0, index=mons, columns=mons)

    for (a, b), row in pair_df.iterrows():
        if a in mons and b in mons:
            value = float(row["teams_used"])
            mat.loc[a, b] = value
            mat.loc[b, a] = value

    return mat


# Plots a heatmap for common pair usage
def plot_pair_heatmap(
    pair_df: pd.DataFrame,
    pokemon_usage_df: pd.DataFrame,
    output_path: str,
    top_n: int = 12,
):
    if pair_df.empty or pokemon_usage_df.empty:
        return

    # Take the top-N Pokémon by overall usage
    top_mons = list(pokemon_usage_df.head(top_n).index)

    mat = build_pair_matrix(pair_df, top_mons)

    plt.figure()
    plt.imshow(mat.values)
    plt.xticks(range(len(top_mons)), top_mons, rotation=45, ha="right")
    plt.yticks(range(len(top_mons)), top_mons)
    plt.title(f"Pair Usage (Teams with Pair) – Top {top_n} Pokémon")
    plt.colorbar(label="Teams with Pair")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Main synergy analysis function
def run_synergy_analysis(
    team_df: pd.DataFrame,
    usage_results: dict,
    output_root: str = "outputs",
):
    synergy_root = os.path.join(output_root, "synergy")
    csv_folder = os.path.join(synergy_root, "csv")
    plots_folder = os.path.join(synergy_root, "plots")

    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # 1) Compute pair usage table
    pair_df = compute_pair_usage(team_df)

    # Save CSV
    pair_csv_path = os.path.join(csv_folder, "pair_usage.csv")
    pair_df.to_csv(pair_csv_path)

    # 2) Plot heatmap for top-N Pokémon
    pokemon_usage_df = usage_results.get("pokemon_usage", pd.DataFrame())
    heatmap_path = os.path.join(plots_folder, "pair_usage_heatmap.png")
    plot_pair_heatmap(pair_df, pokemon_usage_df, heatmap_path, top_n=12)

    # 3) Plot bar chart of top-N pairs
    top_pairs_path = os.path.join(plots_folder, "pair_usage_top_pairs.png")
    plot_top_pairs_bar(pair_df, top_pairs_path, top_n=20)

    # 4) Plot bar chart of bottom-N pairs
    bottom_pairs_path = os.path.join(plots_folder, "pair_usage_bottom_pairs.png")
    plot_bottom_pairs_bar(pair_df, bottom_pairs_path, bottom_n=20)


# Plots a bar chart for most common pairs
def plot_top_pairs_bar(
    pair_df: pd.DataFrame,
    output_path: str,
    top_n: int = 20,
):
    if pair_df.empty:
        return

    # Take top-N pairs
    top = pair_df.head(top_n).copy()

    # Convert MultiIndex (pokemon_1, pokemon_2) into readable labels
    top.index = [f"{a} + {b}" for (a, b) in top.index.to_list()]

    plt.figure()
    top["teams_used"].plot(kind="bar")
    plt.title(f"Top {top_n} Pokémon Pairs (Teams with Pair)")
    plt.ylabel("Teams with Pair")
    plt.xlabel("Pair")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Plots a bar chart for least common pairs
def plot_bottom_pairs_bar(
    pair_df: pd.DataFrame,
    output_path: str,
    bottom_n: int = 20,
):
    if pair_df.empty:
        return

    # Filter pairs that appear at least once, then take the smallest-N
    bottom = pair_df[pair_df["teams_used"] > 0].tail(bottom_n).copy()

    # MultiIndex → readable labels
    bottom.index = [f"{a} + {b}" for (a, b) in bottom.index.to_list()]

    plt.figure()
    bottom["teams_used"].plot(kind="bar")
    plt.title(f"Bottom {bottom_n} Pokémon Pairs (Teams with Pair)")
    plt.ylabel("Teams with Pair")
    plt.xlabel("Pair")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Partner Analysis for a Given Pokémon
# -----------------------------------------------------------------------------


def analyze_pokemon_partners(
    pair_df: pd.DataFrame,
    pokemon_name: str,
    output_root: str = "outputs",
    top_n: int = 10,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(columns=["partner", "teams_used", "usage_rate"])

    target = pokemon_name.strip()

    idx = pair_df.index
    p1 = idx.get_level_values("pokemon_1")
    p2 = idx.get_level_values("pokemon_2")

    mask1 = (p1 == target)
    mask2 = (p2 == target)

    rows = []

    # target as pokemon_1 → partner is pokemon_2
    sub1 = pair_df[mask1]
    for (a, b), row in sub1.iterrows():
        rows.append({
            "partner": b,
            "teams_used": row["teams_used"],
            "usage_rate": row["usage_rate"],
        })

    # target as pokemon_2 → partner is pokemon_1
    sub2 = pair_df[mask2]
    for (a, b), row in sub2.iterrows():
        rows.append({
            "partner": a,
            "teams_used": row["teams_used"],
            "usage_rate": row["usage_rate"],
        })

    if not rows:
        return pd.DataFrame(columns=["partner", "teams_used", "usage_rate"])

    partners_df = pd.DataFrame(rows)

    partners_df = (
        partners_df
        .groupby("partner", as_index=False)
        .agg({"teams_used": "sum", "usage_rate": "mean"})
        .sort_values("teams_used", ascending=False)
    )

    top = partners_df.head(top_n)

    # --- Export CSV and plot into separate folders ---
    partners_root = os.path.join(output_root, "partners")
    csv_folder = os.path.join(partners_root, "csv")
    plots_folder = os.path.join(partners_root, "plots")
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    safe_name = target.replace(" ", "_").replace("/", "_")

    csv_path = os.path.join(csv_folder, f"{safe_name}_partners.csv")
    png_path = os.path.join(plots_folder, f"{safe_name}_partners.png")

    top.to_csv(csv_path, index=False)

    plt.figure()
    top.plot(
        x="partner",
        y="teams_used",
        kind="bar",
        legend=False,
    )
    plt.title(f"Top {top_n} Partners for {target}")
    plt.ylabel("Teams with Pair")
    plt.xlabel("Partner")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    return top


