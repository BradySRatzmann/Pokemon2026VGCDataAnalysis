import streamlit as st
import pandas as pd

from analysis.usage import compute_usage_rates

def render_usage_statistics(results):
    st.header("Usage Statistics")

    # Recalculate usage from records in the selected format so
    # both counts and percentages reflect only that regulation.
    usage_pokemon_df = pokemon_df.copy()

    if "format" in usage_pokemon_df.columns:
        usage_pokemon_df = pokemon_df.copy()

        filtered_usage_results = compute_usage_rates(usage_pokemon_df)
    else:
        st.info(
            "Format information is not in the saved dataset yet. "
            "Run a fresh analysis to enable format filtering."
        )

    filtered_usage_results = compute_usage_rates(usage_pokemon_df)

    usage_mapping = {
        "Pokémon": "pokemon_usage",
        "Items": "item_usage",
        "Natures": "nature_usage",
        "Moves": "move_usage"
    }

    analysis_type = st.selectbox(
        "Select a category:",
        list(usage_mapping.keys())
    )

    selected_data = filtered_usage_results[
        usage_mapping[analysis_type]
    ]

    # Moves use total_uses instead of usage_rate
    value_column = (
        "total_uses" if analysis_type == "Moves"
        else "usage_rate"
    )

    st.subheader(f"{analysis_type} Usage")

    if selected_data.empty:
        st.info("No usage data is available for this format.")
    else:
        st.bar_chart(selected_data[value_column])
        st.dataframe(selected_data)