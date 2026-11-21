import streamlit as st
import pandas as pd

from data_collection.team_parser import get_placement_rank
from visualization.pokemon_images import show_pokemon_image

def render_event_analysis(results):
    st.header("Event Analysis")

    st.write(
        "Explore tournament results, the top 16 players, "
        "their teams, and the most common Pokémon."
    )

    required_columns = [
        "player_name",
        "placement",
        "event_name",
        "source_url"
    ]

    if not all(column in team_df.columns for column in required_columns):
        st.warning(
            "Event metadata is missing. Please run a fresh analysis."
        )

    else:

        event_options = sorted(
            team_df.loc[
                team_df["event_name"] != "Unknown",
                "event_name"
            ].dropna().unique()
        )

        if not event_options:
            st.warning("No event information was found.")

        else:

            selected_event = st.selectbox(
                "Choose an event:",
                event_options,
                key="event_analysis_selection"
            )

            event_teams = team_df[
                team_df["event_name"] == selected_event
                ].copy()

            event_team_ids = event_teams["team_id"].tolist()

            event_pokemon = pokemon_df[
                pokemon_df["team_id"].isin(event_team_ids)
            ].copy()

            # ----------------------------------
            # Event Overview
            # ----------------------------------

            st.divider()
            st.subheader(selected_event)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Teams",
                len(event_teams)
            )

            col2.metric(
                "Players",
                event_teams["player_name"].nunique()
            )

            col3.metric(
                "Unique Pokémon",
                event_pokemon["name"].nunique()
            )

            # ----------------------------------
            # Most Common Pokémon
            # ----------------------------------

            st.divider()
            st.subheader("Most Common Pokémon")

            pokemon_usage = (
                event_pokemon.groupby("name")["team_id"]
                .nunique()
                .sort_values(ascending=False)
            )

            if pokemon_usage.empty:
                st.caption("No Pokémon usage data available.")

            else:

                usage_df = pokemon_usage.head(20).rename(
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
            # Top 16 Players
            # ----------------------------------

            st.divider()
            st.subheader("Top 16 Players")

            # Calculate placement rankings
            leaderboard_df = event_teams.copy()

            leaderboard_df["placement_rank"] = (
                leaderboard_df["placement"].apply(get_placement_rank)
            )

            # Sort players from best to worst placement
            leaderboard_df = leaderboard_df.sort_values(
                ["placement_rank", "player_name"],
                ascending=[True, True]
            ).head(16).reset_index(drop=True)

            st.caption(
                "Each player appears only once, with their recorded "
                "placement and team."
            )

            pokemon_columns = [
                f"pokemon_{i}" for i in range(1, 7)
                if f"pokemon_{i}" in event_teams.columns
            ]

            # Display each player and their team
            for rank, (_, player) in enumerate(
                    leaderboard_df.iterrows(),
                    start=1
            ):

                with st.expander(
                        f"#{rank} - {player['player_name']} "
                        f"({player['placement']})"
                ):

                    st.write(
                        f"**Placement:** {player['placement']}"
                    )

                    if pd.notna(player.get("source_url")):
                        st.markdown(
                            f"[Open Original Poképaste]({player['source_url']})"
                        )

                    # Display the player's six Pokémon
                    sprite_columns = st.columns(6)

                    for i, column in enumerate(pokemon_columns):
                        pokemon = player[column]

                        if pd.notna(pokemon) and str(pokemon).strip():
                            with sprite_columns[i]:
                                show_pokemon_image(
                                    pokemon,
                                    width=75
                                )
                                st.caption(str(pokemon).title())

            # Display a compact leaderboard table
            st.subheader("Leaderboard Data")

            st.dataframe(
                leaderboard_df[
                    [
                        "player_name",
                        "placement"
                    ]
                ].rename(columns={
                    "player_name": "Player",
                    "placement": "Placement"
                }),
                width="stretch",
                hide_index=True
            )