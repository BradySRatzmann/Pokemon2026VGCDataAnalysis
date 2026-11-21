import streamlit as st
import pandas as pd

from data_collection.team_parser import get_placement_rank
from visualization.pokemon_images import show_pokemon_image

def render_player_analysis(results):
    st.header("Player Analysis")

    st.write(
        "Explore individual players, their teams, "
        "most-used Pokémon, and competitive placements."
    )

    required_columns = [
        "player_name",
        "placement",
        "event_name",
        "source_url"
    ]

    if not all(column in team_df.columns for column in required_columns):
        st.warning(
            "Player metadata is missing. Please run a fresh analysis."
        )

    else:

        player_options = sorted(
            player for player in team_df["player_name"].dropna().unique()
            if player != "Unknown"
        )

        if not player_options:
            st.warning("No player information was found.")

        else:

            selected_player = st.selectbox(
                "Choose a player:",
                player_options,
                key="player_analysis_selection"
            )

            player_teams = team_df[
                team_df["player_name"] == selected_player
                ].copy()

            player_team_ids = player_teams["team_id"].tolist()

            player_pokemon = pokemon_df[
                pokemon_df["team_id"].isin(player_team_ids)
            ].copy()

            # ----------------------------------
            # Player Overview
            # ----------------------------------

            st.divider()
            st.subheader(selected_player)

            col1, col2, col3 = st.columns(3)

            best_placement = player_teams.copy()
            best_placement["placement_rank"] = (
                best_placement["placement"].apply(get_placement_rank)
            )

            best_placement = best_placement.sort_values("placement_rank").iloc[0]

            col1.metric(
                "Top Placement",
                best_placement["placement"]
            )

            col2.metric(
                "Events",
                player_teams["event_name"].nunique()
            )

            col3.metric(
                "Unique Pokémon",
                player_pokemon["name"].nunique()
            )

            # ----------------------------------
            # Most Used Pokémon
            # ----------------------------------

            st.divider()
            st.subheader("Most Used Pokémon")

            pokemon_usage = (
                player_pokemon.groupby("name")["team_id"]
                .nunique()
                .sort_values(ascending=False)
            )

            if pokemon_usage.empty:
                st.caption("No Pokémon usage data available.")

            else:

                usage_df = pokemon_usage.head(10).rename(
                    "Teams Used"
                ).to_frame()

                st.bar_chart(usage_df)

                sprite_columns = st.columns(5)

                for i, pokemon in enumerate(usage_df.index):
                    with sprite_columns[i % 5]:
                        show_pokemon_image(pokemon, width=75)
                        st.caption(pokemon.title())

                st.dataframe(
                    usage_df,
                    width="stretch"
                )

            # ----------------------------------
            # Common Placements
            # ----------------------------------

            st.divider()
            st.subheader("Common Placements")

            placement_counts = (
                player_teams["placement"]
                .fillna("Unknown")
                .value_counts()
            )

            if placement_counts.empty:
                st.caption("No placement information available.")

            else:
                st.bar_chart(placement_counts)
                st.dataframe(
                    placement_counts.rename("Events").to_frame(),
                    width="stretch"
                )

            # ----------------------------------
            # Player Teams
            # ----------------------------------

            st.divider()
            st.subheader("Player Teams")

            st.caption(
                f"Showing {len(player_teams)} team(s) found for "
                f"{selected_player}."
            )

            pokemon_columns = [
                f"pokemon_{i}" for i in range(1, 7)
                if f"pokemon_{i}" in player_teams.columns
            ]

            for _, team in player_teams.iterrows():

                with st.expander(
                        f"{team['event_name']} - {team['placement']}"
                ):

                    st.write(
                        f"**Event:** {team['event_name']}"
                    )

                    st.write(
                        f"**Placement:** {team['placement']}"
                    )

                    if pd.notna(team.get("source_url")):
                        st.markdown(
                            f"[Open Original Poképaste]({team['source_url']})"
                        )

                    sprite_columns = st.columns(6)

                    for i, column in enumerate(pokemon_columns):
                        pokemon = team[column]

                        if pd.notna(pokemon) and str(pokemon).strip():
                            with sprite_columns[i]:
                                show_pokemon_image(pokemon, width=75)
                                st.caption(str(pokemon).title())
