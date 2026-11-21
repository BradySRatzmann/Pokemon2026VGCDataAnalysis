import streamlit as st
import pandas as pd
def render_dashboard(results):
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Teams",
        team_df["team_id"].nunique()
    )

    col2.metric(
        "Unique Pokémon",
        pokemon_df["name"].nunique()
    )

    col3.metric(
        "Total Pokémon Entries",
        len(pokemon_df)
    )

    st.divider()

    st.subheader("Most Used Pokémon")

    usage_df = (
        usage_results["pokemon_usage"]
        .reset_index()
        .head(10)
    )

    # Sort Pokémon alphabetically to match the chart order
    usage_df = usage_df.sort_values(
        by="name",
        key=lambda col: col.str.lower()
    ).reset_index(drop=True)

    st.bar_chart(
        usage_df,
        x="name",
        y="usage_rate"
    )
    # Display Pokémon sprites aligned underneath their respective bars
    sprite_columns = st.columns([0.72] + [1] * 10)

    for i, (_, pokemon) in enumerate(usage_df.iterrows()):
        with sprite_columns[i + 1]:
            show_pokemon_image(pokemon['name'])