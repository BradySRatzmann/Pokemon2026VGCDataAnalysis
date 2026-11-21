import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os

# Usage Rate Analysis
# -----------------------------------------------------------------------------


# Compute basic usage rates for Pokémon, items, tera types, and moves
def compute_usage_rates(pokemon_df):
    results = {}

    total_teams = pokemon_df["team_id"].nunique()

    if total_teams == 0:
        empty_usage = pd.DataFrame(columns=["teams_used", "usage_rate"])
        return {
            "pokemon_usage": empty_usage.copy(),
            "item_usage": empty_usage.copy(),
            "nature_usage": empty_usage.copy(),
            "move_usage": pd.DataFrame(columns=["total_uses"])
        }

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

    # --- Nature Usage ---
    nature_usage = (
        pokemon_df.groupby("nature")["team_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("teams_used")
        .to_frame()
    )
    nature_usage["usage_rate"] = nature_usage["teams_used"] / total_teams * 100
    results["nature_usage"] = nature_usage

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

    # --- Nature usage ---
    if "nature_usage" in usage_results:
        df = usage_results["nature_usage"].head(top_n)
        plt.figure()
        df["usage_rate"].plot(kind="bar")
        plt.title(f"Top {top_n} Nature Usage Rate")
        plt.ylabel("Usage Rate (%)")
        plt.xlabel("Nature")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "nature_usage_top.png"))
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

    # --- Nature ---
    nature_counts = sub["nature"].value_counts().head(top_n)
    nature_counts.to_csv(os.path.join(csv_folder, f"{safe_name}_natures.csv"))

    plt.figure()
    nature_counts.plot(kind="bar")
    plt.title(f"Top Natures for {target}")
    plt.ylabel("Usage Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f"{safe_name}_natures.png"))
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