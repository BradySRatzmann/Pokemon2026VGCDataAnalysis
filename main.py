# === Imports ===
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer, Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, top_k_accuracy_score


# Initial Data Fetching and Tokenizing Functions
# -----------------------------------------------------------------------------


# Function to get the URLs from a text file
def read_data(filename: str):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]


# Asynchronous helper function to fetch a single Poképaste page
async def fetch_page(session, url: str):
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, "lxml")  # Faster parser
                paragraphs = soup.find_all("article")
                return [para.get_text() for para in paragraphs]
            else:
                print(f"⚠️ Failed to fetch {url} (status {resp.status})")
                return []
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return []


# Function to concurrently fetch multiple Poképaste pages
async def read_all_urls(urls: list[str]):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_page(session, url) for url in urls]
        return await asyncio.gather(*tasks)


# Function to remove unwanted words from a list of words
def word_removal(words: list[str], remove: list[str]):
    return [word for word in words if word not in remove]


# Function to clean and combine certain words into compound terms
def cleaned_or_Combined(words: list[str], remove: list[str]):
    cleaned = []
    i = 0

    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    while i < len(words):
        if words[i] in remove or is_int(words[i]):
            i += 1
            continue

        combined = [words[i]]
        j = i + 1
        while j < len(words) and words[j] not in remove and not is_int(words[j]):
            combined.append(words[j])
            j += 1

        cleaned.append("-".join(combined))
        i = j
    return cleaned


# Function to tokenize a piece of text formatted for the pokepast.es website
def tokenize(sentence: str):
    # Detects if one optional attribute is missing and adds "none" if so
    if "@" in sentence:
        before_at, after_at = sentence.split("@", 1)

        if after_at.strip() == "":
            sentence = before_at + "@ none"

    sentence = sentence.lower()
    remove_chars = [".", ",", "?", "!", "\n", ":", ";", "(", ")", "/"]
    remove_words = ["@", "-", "ability", "tera", "type", "level", "shiny", "yes",
                    "nature", "evs", "ivs", "hp", "atk", "def", "spa", "spd", "spe"]

    # Remove unwanted punctuation
    cleaned = "".join(char for char in sentence if char not in remove_chars)

    # Split into words
    words = cleaned.split()

    # Combine multiword terms
    words = cleaned_or_Combined(words, remove_words)

    # Normalize special cases
    words = ["tera-blast" if w == "blast" else w for w in words]

    # Remove separator words
    return word_removal(words, remove_words)



# Build DataFrames based on tokenized data
# -----------------------------------------------------------------------------


# Formatting for Pokémon DataFrame rows
def data_to_row(tokens: list[str]):
    columns = [
        "name",
        "item",
        "ability",
        "tera_type",
        "move_1",
        "move_2",
        "move_3",
        "move_4",
    ]

    # Case 1: Full 8 tokens (item present)
    if len(tokens) == 8:
        fixed = tokens

    # Case 2: 7 tokens (item missing)
    elif len(tokens) == 7:
        # Insert literal "none" in item slot
        fixed = [tokens[0], "none"] + tokens[1:]

    # Case 3: Unexpected cases — pad or truncate safely
    else:
        fixed = (tokens + ["none"] * 8)[:8]

    return dict(zip(columns, fixed))


# Combines the original URL with its tokenized Pokémon data
def build_team_tokenized(data_points, paragraphs_list):
    return [
        (url, [tokenize(p) for p in paragraphs])
        for url, paragraphs in zip(data_points, paragraphs_list)
    ]


# Build team-level DataFrame with one row per team.
# Each team gets a unique ID, the URL, and the six Pokémon names.
def build_team_dataframe(team_tokenized):
    rows = []

    for team_id, (url, pokemons) in enumerate(team_tokenized, start=1):
        row = {
            "team_id": team_id,
            "source_url": url,
        }

        for i, pkm in enumerate(pokemons, start=1):
            name = pkm[0] if pkm else None
            row[f"pokemon_{i}"] = name

        rows.append(row)

    return pd.DataFrame(rows)


# Usage Rate Analysis
# -----------------------------------------------------------------------------


# Compute basic usage rates for Pokémon, items, tera types, and moves
def compute_usage_rates(pokemon_df):
    results = {}

    total_teams = pokemon_df["team_id"].nunique()

    # --- Pokémon Usage Rate ---
    usage = (
        pokemon_df.groupby("name")["team_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("teams_used")
        .to_frame()
    )
    usage["usage_rate"] = usage["teams_used"] / total_teams * 100
    results["pokemon_usage"] = usage

    # --- Item Usage Rate ---
    item_usage = (
        pokemon_df.groupby("item")["team_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("teams_used")
        .to_frame()
    )
    item_usage["usage_rate"] = item_usage["teams_used"] / total_teams * 100
    results["item_usage"] = item_usage

    # --- Tera Type Usage ---
    tera_usage = (
        pokemon_df.groupby("tera_type")["team_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("teams_used")
        .to_frame()
    )
    tera_usage["usage_rate"] = tera_usage["teams_used"] / total_teams * 100
    results["tera_usage"] = tera_usage

    # --- Move Usage ---
    move_cols = ["move_1", "move_2", "move_3", "move_4"]
    move_df = (
        pokemon_df[move_cols]
        .melt(value_name="move")
        .dropna()
    )
    move_usage = (
        move_df.groupby("move")
        .size()
        .sort_values(ascending=False)
        .rename("total_uses")
        .to_frame()
    )
    results["move_usage"] = move_usage

    return results


# Plots basic bar charts for usage tables
def plot_usage_tables(usage_results: dict, output_folder: str, top_n: int = 20):
    os.makedirs(output_folder, exist_ok=True)

    # --- Pokémon usage ---
    if "pokemon_usage" in usage_results:
        df = usage_results["pokemon_usage"].head(top_n)
        plt.figure()
        df["usage_rate"].plot(kind="bar")
        plt.title(f"Top {top_n} Pokémon Usage Rate")
        plt.ylabel("Usage Rate (%)")
        plt.xlabel("Pokémon")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "pokemon_usage_top.png"))
        plt.close()

    # --- Item usage ---
    if "item_usage" in usage_results:
        df = usage_results["item_usage"].head(top_n)
        plt.figure()
        df["usage_rate"].plot(kind="bar")
        plt.title(f"Top {top_n} Item Usage Rate")
        plt.ylabel("Usage Rate (%)")
        plt.xlabel("Item")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "item_usage_top.png"))
        plt.close()

    # --- Tera type usage ---
    if "tera_usage" in usage_results:
        df = usage_results["tera_usage"].head(top_n)
        plt.figure()
        df["usage_rate"].plot(kind="bar")
        plt.title(f"Top {top_n} Tera Type Usage Rate")
        plt.ylabel("Usage Rate (%)")
        plt.xlabel("Tera Type")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "tera_usage_top.png"))
        plt.close()

    # --- Move usage ---
    if "move_usage" in usage_results:
        df = usage_results["move_usage"].head(top_n)
        plt.figure()
        df["total_uses"].plot(kind="bar")
        plt.title(f"Top {top_n} Move Usage (Total Uses)")
        plt.ylabel("Total Uses")
        plt.xlabel("Move")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "move_usage_top.png"))
        plt.close()


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


# Attribute Analysis for a Given Pokémon
# -----------------------------------------------------------------------------

def analyze_pokemon_attributes(
    pokemon_df: pd.DataFrame,
    pokemon_name: str,
    output_root: str = "outputs",
    top_n: int = 10,
):
    target = pokemon_name.strip().lower()

    sub = pokemon_df[pokemon_df["name"] == target]
    if sub.empty:
        return

    attr_root = os.path.join(output_root, "attributes")
    csv_folder = os.path.join(attr_root, "csv")
    plots_folder = os.path.join(attr_root, "plots")
    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    safe_name = target.replace(" ", "_").replace("/", "_")

    # --- Ability ---
    ability_counts = sub["ability"].value_counts().head(top_n)
    ability_counts.to_csv(os.path.join(csv_folder, f"{safe_name}_abilities.csv"))

    plt.figure()
    ability_counts.plot(kind="bar")
    plt.title(f"Top Abilities for {target}")
    plt.ylabel("Usage Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"{safe_name}_abilities.png"))
    plt.close()

    # --- Item ---
    item_counts = sub["item"].value_counts().head(top_n)
    item_counts.to_csv(os.path.join(csv_folder, f"{safe_name}_items.csv"))

    plt.figure()
    item_counts.plot(kind="bar")
    plt.title(f"Top Items for {target}")
    plt.ylabel("Usage Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"{safe_name}_items.png"))
    plt.close()

    # --- Tera Type ---
    tera_counts = sub["tera_type"].value_counts().head(top_n)
    tera_counts.to_csv(os.path.join(csv_folder, f"{safe_name}_tera_types.csv"))

    plt.figure()
    tera_counts.plot(kind="bar")
    plt.title(f"Top Tera Types for {target}")
    plt.ylabel("Usage Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"{safe_name}_tera_types.png"))
    plt.close()

    # --- Moves ---
    move_cols = ["move_1", "move_2", "move_3", "move_4"]
    moves = sub[move_cols].melt(value_name="move")["move"]
    move_counts = moves.value_counts().head(top_n)
    move_counts.to_csv(os.path.join(csv_folder, f"{safe_name}_moves.csv"))

    plt.figure()
    move_counts.plot(kind="bar")
    plt.title(f"Top Moves for {target}")
    plt.ylabel("Usage Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"{safe_name}_moves.png"))
    plt.close()


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

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_red)

        score = silhouette_score(X_red, labels) if len(set(labels)) > 1 else -1.0
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

    # Cluster summaries: top mons by within-cluster frequency
    summaries = []
    X_arr = X  # numpy array already

    for c in range(best_k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        counts = X_arr[idx].sum(axis=0)  # frequency within cluster
        top_idx = np.argsort(-counts)[:15]
        top_mons = [f"{vocab[i]}({int(counts[i])})" for i in top_idx if counts[i] > 0]

        summaries.append({
            "cluster": int(c),
            "num_teams": int(len(idx)),
            "top_mons": "; ".join(top_mons),
        })

    pd.DataFrame(summaries).sort_values("num_teams", ascending=False).to_csv(
        os.path.join(out_dir, "cluster_summaries.csv"), index=False
    )

    return best_k, float(best_score)

# Pick 6th team member based on first 5 members
# -----------------------------------------------------------------------------

def build_sixth_mon_dataset(team_df: pd.DataFrame):

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


# Token Analysis Utilities
# -----------------------------------------------------------------------------


# Helper function to print tokenized data
def print_tokenized(tokenized_sentence: list[list[str]]):
    for sentence in tokenized_sentence:
        print(sentence)


def analyze_tokens(tokenized_data: list[list[str]]):
    all_tokens = [token for sentence in tokenized_data for token in sentence]
    df = pd.DataFrame(all_tokens, columns=["token"])

    print("\n--- Summary statistics ---")
    print(f"Total tokens: {len(df)}")
    print(f"Unique tokens: {df['token'].nunique()}")

    token_counts = df["token"].value_counts()
    print("\n--- Top 10 most common tokens ---")
    print(token_counts.head(10))

    df["length"] = df["token"].apply(len)
    df["first_letter"] = df["token"].str[0]

    print("\n--- Average token length by first letter ---")
    print(df.groupby("first_letter")["length"].mean())

    lengths = df["length"].to_numpy()
    print("\n--- NumPy stats on token lengths ---")
    print(f"Mean: {np.mean(lengths):.2f}")
    print(f"Std Dev: {np.std(lengths):.2f}")
    print(f"Max: {np.max(lengths)}")
    print(f"Min: {np.min(lengths)}")




# Main Execution Flow
# -----------------------------------------------------------------------------


def main():
    data_points = read_data("data.txt")
    print(f"⏳ Fetching {len(data_points)} Poképaste pages concurrently...\n")

    # Fetch all pages asynchronously
    paragraphs_list = asyncio.run(read_all_urls(data_points))

    # Build grouped team data mapping URLs → their Pokémon lists
    team_tokenized = build_team_tokenized(data_points, paragraphs_list)

    # Build Pokémon-level DataFrame
    pokemon_rows = []
    for team_id, (url, pokemons) in enumerate(team_tokenized, start=1):
        for tokens in pokemons:
            # Skip completely empty slots (no Pokémon parsed)
            if not tokens:
                continue

            row = data_to_row(tokens)
            row["team_id"] = team_id
            row["source_url"] = url
            pokemon_rows.append(row)

    pokemon_df = pd.DataFrame(pokemon_rows)
    print("\n--- Pokémon DataFrame Preview ---")
    print(pokemon_df.head())

    # Build team-level DataFrame
    team_df = build_team_dataframe(team_tokenized)
    print("\n--- Team DataFrame Preview ---")
    print(team_df.head())

    # === Save Pokémon + Team DataFrames into organized folders ===
    output_root = "outputs"
    data_folder = os.path.join(output_root, "data")
    os.makedirs(data_folder, exist_ok=True)

    pokemon_df.to_csv(os.path.join(data_folder, "pokemon_structured.csv"), index=False)
    team_df.to_csv(os.path.join(data_folder, "teams.csv"), index=False)

    # === Usage Rate Analysis ===
    usage_results = compute_usage_rates(pokemon_df)

    print("\n--- Top Pokémon Usage ---")
    print(usage_results["pokemon_usage"].head(10))

    print("\n--- Top Item Usage ---")
    print(usage_results["item_usage"].head(10))

    print("\n--- Top Tera-Type Usage ---")
    print(usage_results["tera_usage"].head(10))

    print("\n--- Top Moves ---")
    print(usage_results["move_usage"].head(10))

    # === Save usage output into organized folders inside outputs/ ===
    output_root = "outputs"
    usage_root = os.path.join(output_root, "usage")
    csv_folder = os.path.join(usage_root, "csv")
    plots_folder = os.path.join(usage_root, "plots")

    os.makedirs(csv_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # Save usage CSV files
    for name, df_out in usage_results.items():
        df_out.to_csv(os.path.join(csv_folder, f"{name}.csv"))

    # Save usage plots
    plot_usage_tables(usage_results, plots_folder, top_n=20)

    # === Pair (Synergy) Analysis ===
    run_synergy_analysis(team_df, usage_results, output_root=output_root)

    # === Partner Analysis for Selected Pokémon ===
    # Example list — edit or expand as you wish
    pokemon_to_check = ["gholdengo", "kingambit", "dragonite", "incineroar", "amoonguss","not real test"]

   
    pair_df_path = os.path.join("outputs", "synergy", "csv", "pair_usage.csv")
    pair_df = pd.read_csv(pair_df_path, index_col=[0, 1])

    # Analyze partners and attributes for each Pokémon in the list
    for mon in pokemon_to_check:
        analyze_pokemon_partners(pair_df, mon, output_root="outputs", top_n=10)

    for mon in pokemon_to_check:
        analyze_pokemon_attributes(pokemon_df, mon, output_root="outputs", top_n=10)

    print(f"⏳ Creating cluster Summaries of common team archetypes...\n")
    # output cvs file with plots of cluster summaries of common archetypes
    best_k, best_score = run_archetype_clustering(
        team_df,
        output_root="outputs",
        k_min=4,
        k_max=20,
        svd_components=50,
    )
    print(f"✅ archetypes clustering: best_k={best_k}, silhouette={best_score:.3f}\n")

    print(f"⏳ trains a model to recommend a new Pokemon given Pokemon already on the team\n")
    # trains a model to recommend a new mon given mons already on the team
    clf, mlb, metrics = train_sixth_mon_model(team_df, random_state=0)
    print("✅ new Pokémon model metrics:", metrics)

    example_team5 = ["pelipper","archaludon","basculegion","incineroar","amoonguss"]
    recs = recommend_sixth_mon(clf, mlb, example_team5, top_n=10)
    print("Top recommendations for", example_team5)
    for mon, p in recs:
        print(f"  {mon:15s}  {p:.3f}")


if __name__ == "__main__":
    main()
