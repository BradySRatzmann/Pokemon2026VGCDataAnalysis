import os
import pandas as pd
import numpy as np
import asyncio

from data_collection.team_parser import (
    parse_team_metadata,
    get_placement_rank,
    word_removal,
    cleaned_or_Combined,
    tokenize,
    data_to_row,
    build_team_tokenized,
    build_team_dataframe
)
from analysis.partners import (
    compute_pair_usage,
    analyze_pokemon_partners,
    run_synergy_analysis
)

from analysis.prediction import (
    build_sixth_mon_dataset,
    train_sixth_mon_model,
    recommend_sixth_mon
)

from data_collection.data_loader import read_data
from data_collection.pokepaste import read_all_urls

from analysis.usage import (
    compute_usage_rates,
    plot_usage_tables,
    analyze_pokemon_attributes
)

from analysis.partners import (
    analyze_pokemon_partners,
    run_synergy_analysis
)

from analysis.prediction import (
    build_sixth_mon_dataset,
    train_sixth_mon_model
)

from analysis.archetypes import run_archetype_clustering

def run_analysis(data_points=None):
    if data_points is None:
        data_points = read_data("data.txt")
    print(f"⏳ Fetching {len(data_points)} Poképaste pages concurrently...\n")

    # Fetch all pages asynchronously
    paragraphs_list = asyncio.run(read_all_urls(data_points))

    # Build grouped team data mapping URLs → their Pokémon lists
    # Build grouped team data mapping URLs → their Pokémon lists
    team_tokenized = build_team_tokenized(data_points, paragraphs_list)

    # Keep only teams that placed Top 256 or better
    team_tokenized = [
        (url, pokemons, metadata)
        for url, pokemons, metadata in team_tokenized
        if get_placement_rank(metadata["placement"]) <= 256
    ]

    print(
        f"✅ Keeping {len(team_tokenized)} teams "
        f"that placed Top 256 or better."
    )

    # Build Pokémon-level DataFrame
    pokemon_rows = []
    for team_id, (url, pokemons, metadata) in enumerate(team_tokenized, start=1):
        for tokens in pokemons:
            # Skip completely empty slots (no Pokémon parsed)
            if not tokens:
                continue

            row = data_to_row(tokens)
            row["team_id"] = team_id
            row["source_url"] = url
            pokemon_rows.append(row)

    pokemon_df = pd.DataFrame(pokemon_rows)

    # Build team-level DataFrame
    team_df = build_team_dataframe(team_tokenized)

    # === Save Pokémon + Team DataFrames into organized folders ===
    output_root = "outputs"
    data_folder = os.path.join(output_root, "data")
    os.makedirs(data_folder, exist_ok=True)

    pokemon_df.to_csv(os.path.join(data_folder, "pokemon_structured.csv"), index=False)
    team_df.to_csv(os.path.join(data_folder, "teams.csv"), index=False)

    # === Usage Rate Analysis ===
    usage_results = compute_usage_rates(pokemon_df)

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


    print(f"⏳ trains a model to recommend a new Pokemon given Pokemon already on the team\n")
    # # trains a model to recommend a new mon given mons already on the team
    clf, mlb, metrics = train_sixth_mon_model(team_df, random_state=0)
    print("✅ new Pokémon model metrics:", metrics)


    # === Prepare results for the UI ===

    pair_df = compute_pair_usage(team_df)


    # === Team Archetype Clustering ===
    print("⏳ Analyzing team archetypes...")

    best_k, best_score, archetype_df, team_clusters = run_archetype_clustering(
        team_df,
        output_root=output_root,
        k_min=4,
        k_max=20,
        svd_components=50,
    )

    pokemon_to_check = ["chandelure","blastoise-mega"]
    recs = recommend_sixth_mon(clf, mlb, pokemon_to_check, top_n=10)
    print("Top recommendations for", pokemon_to_check)
    for mon, p in recs:
         print(f"  {mon:15s}  {p:.3f}")

    print(
        f"✅ Archetype clustering completed: "
        f"best_k={best_k}, silhouette={best_score:.3f}"
    )
    # # === Return results to the Streamlit UI ===
    return {
        "pokemon_df": pokemon_df,
        "team_df": team_df,
        "usage_results": usage_results,
        "pair_df": pair_df,
        "archetype_df": archetype_df,
        "team_clusters": team_clusters,
        "best_k": best_k,
        "best_score": best_score,
        "sixth_mon_model": clf,
        "sixth_mon_mlb": mlb,
        "sixth_mon_metrics": metrics
    }

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