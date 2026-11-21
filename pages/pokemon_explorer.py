import streamlit as st
import pandas as pd

from analysis.partners import analyze_pokemon_partners
from visualization.pokemon_images import show_pokemon_image
from visualization.item_images import show_item_image

def render_pokemon_explorer(results):
    st.header("Pokémon Explorer")

    st.write(
        "Select a Pokémon to explore its usage, common sets, "
        "and the teams it appears on."
    )

    pokemon_options = sorted(
        pokemon_df["name"].dropna().unique()
    )

    if not pokemon_options:
        st.warning("No Pokémon were found in the current dataset.")
    else:
        selected_pokemon = st.selectbox(
            "Choose a Pokémon:",
            pokemon_options,
            key="explorer_pokemon"
        )

        selected_rows = pokemon_df[
            pokemon_df["name"] == selected_pokemon
            ].copy()

        # Pokémon image and basic usage information
        image_col, info_col = st.columns([1, 4])

        with image_col:
            show_pokemon_image(selected_pokemon, width=200)

        pokemon_usage = usage_results.get(
            "pokemon_usage", pd.DataFrame()
        )

        with info_col:
            st.subheader(selected_pokemon.title())

            if selected_pokemon in pokemon_usage.index:
                usage_row = pokemon_usage.loc[selected_pokemon]

                # Find teams featuring this Pokémon
                pokemon_columns = [
                    f"pokemon_{i}" for i in range(1, 7)
                    if f"pokemon_{i}" in team_df.columns
                ]

                team_mask = team_df[pokemon_columns].eq(
                    selected_pokemon
                ).any(axis=1)

                pokemon_teams = team_df.loc[team_mask].copy()

                # Find the highest placement
                pokemon_teams["placement_rank"] = (
                    pokemon_teams["placement"].apply(get_placement_rank)
                )

                best_team = pokemon_teams.sort_values(
                    "placement_rank"
                ).iloc[0]

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                metric_col1.metric(
                    "Teams Used",
                    int(usage_row["teams_used"])
                )

                metric_col2.metric(
                    "Usage Rate",
                    f'{usage_row["usage_rate"]:.1f}%'
                )

                metric_col3.metric(
                    "Best Placement",
                    best_team["placement"]
                )

                pokemon_rank = pokemon_usage.index.get_loc(
                    selected_pokemon
                ) + 1

                st.caption(
                    f"Usage ranking: #{pokemon_rank} "
                    f"out of {len(pokemon_usage)} Pokémon"
                )

                st.caption(
                    f"Achieved by {best_team['player_name']} "
                    f"at {best_team['event_name']}."
                )

        st.divider()

        # Common attributes are calculated from the already-loaded
        # Pokémon rows; no additional scraping or analysis run is needed.
        attribute_columns = st.columns(3)

        attribute_sections = [
            ("Most Common Items", "item"),
            ("Most Common Abilities", "ability"),
            ("Most Common Natures", "nature")
        ]

        for column, (section_title, field) in zip(
                attribute_columns, attribute_sections
        ):
            with column:
                st.subheader(section_title)
                counts = (
                    selected_rows[field]
                    .dropna()
                    .astype(str)
                )
                counts = counts[
                    (counts != "") & (counts.str.lower() != "none")
                    ].value_counts().head(5)

                if counts.empty:
                    st.caption("No data available.")
                else:
                    if field == "item":
                        item_display = counts.rename("Uses").reset_index()
                        item_display.columns = ["Item", "Uses"]

                        # Generate item image URLs, including fallback images
                        item_display["Image"] = item_display["Item"].apply(
                            get_item_image
                        )

                        # Put the image column first
                        item_display = item_display[["Image", "Item", "Uses"]]

                        st.dataframe(
                            item_display,
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "Image": st.column_config.ImageColumn(
                                    "Image",
                                    width="small"
                                )
                            }
                        )

                    else:
                        st.dataframe(
                            counts.rename("Uses").to_frame(),
                            width="stretch"
                        )

        st.subheader("Most Common Moves")

        move_columns = ["move_1", "move_2", "move_3", "move_4"]
        available_move_columns = [
            column for column in move_columns
            if column in selected_rows.columns
        ]

        if available_move_columns:
            move_counts = (
                selected_rows[available_move_columns]
                .melt(value_name="move")["move"]
                .dropna()
                .astype(str)
            )
            move_counts = move_counts[
                (move_counts != "") & (move_counts.str.lower() != "none")
                ].value_counts().head(10)

            if move_counts.empty:
                st.caption("No move data available.")
            else:
                st.bar_chart(move_counts)
                st.dataframe(
                    move_counts.rename("Uses").to_frame(),
                    width="stretch"
                )
        else:
            st.caption("Move information is not present in this dataset.")

        st.divider()
        st.subheader("Teams Featuring This Pokémon")

        team_columns = [
            f"pokemon_{i}" for i in range(1, 7)
            if f"pokemon_{i}" in team_df.columns
        ]

        if team_columns:
            team_mask = team_df[team_columns].eq(selected_pokemon).any(axis=1)
            matching_teams = team_df.loc[team_mask]

            st.caption(
                f"Found in {len(matching_teams)} teams in this dataset."
            )

            if matching_teams.empty:
                st.caption("No matching team records were found.")
            else:
                # Include player, placement, and event information
                display_columns = [
                    column for column in [
                                             "player_name",
                                             "placement",
                                             "event_name"
                                         ] + team_columns
                    if column in matching_teams.columns
                ]

                st.dataframe(
                    matching_teams[display_columns],
                    width="stretch",
                    hide_index=True
                )
        else:
            st.caption("Team membership data is not available.")

        # ------------------------------------------
        # Pokémon Partner Explorer
        # ------------------------------------------

        st.divider()
        st.subheader("Pokémon Partner Explorer")

        # Use the same selected Pokémon as the rest of this page
        pair_df = results["pair_df"]

        if pair_df.empty:
            st.warning("No Pokémon pair data was found.")

        else:
            partners_df = analyze_pokemon_partners(
                pair_df,
                selected_pokemon,
                top_n=10,
            )

            if partners_df.empty:

                st.warning(
                    f"No partner data found for {selected_pokemon}."
                )

            else:

                st.write(
                    f"Most common partners for **{selected_pokemon.title()}**:"
                )

                # Display partner images
                partner_columns = st.columns(5)

                for i, (_, partner) in enumerate(partners_df.iterrows()):
                    with partner_columns[i % 5]:
                        show_pokemon_image(
                            partner["partner"],
                            width=80
                        )

                        st.caption(partner["partner"].title())

                        st.caption(
                            f'{int(partner["teams_used"])} teams'
                        )

                # Display partner chart
                st.bar_chart(
                    partners_df.set_index("partner")["teams_used"]
                )

                # Display partner table
                st.dataframe(
                    partners_df,
                    width="stretch",
                    hide_index=True
                )