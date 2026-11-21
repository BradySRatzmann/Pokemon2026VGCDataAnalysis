def render_team_database(results):
    st.header("Team Database")

    st.write(
        "Browse the complete team dataset and filter teams "
        "by format, player, placement, event, or Pokémon."
    )

    # Pokémon columns
    pokemon_columns = [
        f"pokemon_{i}" for i in range(1, 7)
        if f"pokemon_{i}" in team_df.columns
    ]

    # Required metadata columns
    metadata_columns = [
        "player_name",
        "placement",
        "event_name"
    ]

    # Check whether the loaded dataset contains metadata
    missing_columns = [
        column for column in metadata_columns
        if column not in team_df.columns
    ]

    if missing_columns:
        st.warning(
            "Player, placement, and event information is missing "
            "from the currently loaded dataset. Please run a fresh "
            "analysis to load the updated team metadata."
        )

    elif not pokemon_columns:
        st.warning("No Pokémon columns were found in the team dataset.")

    else:

        # ------------------------------------------
        # Filter Options
        # ------------------------------------------

        st.subheader("Filter Teams")

        col1, col2 = st.columns(2)

        with col1:
            if "format" in team_df.columns:
                selected_formats = st.multiselect(
                    "Format",
                    sorted(team_df["format"].dropna().unique()),
                    placeholder="Select formats..."
                )
            else:
                selected_formats = []
                st.info(
                    "Format information is not in the saved dataset yet. "
                    "Run a fresh analysis to enable format filtering."
                )

            selected_players = st.multiselect(
                "Player",
                sorted(team_df["player_name"].dropna().unique()),
                placeholder="Select players..."
            )

            selected_placements = st.multiselect(
                "Placement",
                sorted(team_df["placement"].dropna().unique()),
                placeholder="Select placements..."
            )

        with col2:
            selected_events = st.multiselect(
                "Event",
                sorted(team_df["event_name"].dropna().unique()),
                placeholder="Select events..."
            )

            # Get every unique Pokémon across all six slots
            all_pokemon = sorted(
                set(
                    str(pokemon)
                    for pokemon in team_df[pokemon_columns].values.flatten()
                    if pd.notna(pokemon) and str(pokemon).strip() != ""
                )
            )

            selected_pokemon = st.multiselect(
                "Pokémon",
                all_pokemon,
                placeholder="Select Pokémon..."
            )

        st.caption(
            "Leave a filter empty to include all values. "
            "When multiple Pokémon are selected, teams must contain "
            "all selected Pokémon."
        )

        # ------------------------------------------
        # Apply Filters
        # ------------------------------------------

        # Keep one team per player, event, and format.
        dedupe_columns = ["player_name", "event_name"]
        if "format" in team_df.columns:
            dedupe_columns.append("format")

        filtered_df = team_df.drop_duplicates(
            subset=dedupe_columns,
            keep="first"
        ).copy()

        if selected_formats and "format" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["format"].isin(selected_formats)
            ]

        if selected_players:
            filtered_df = filtered_df[
                filtered_df["player_name"].isin(selected_players)
            ]

        if selected_placements:
            filtered_df = filtered_df[
                filtered_df["placement"].isin(selected_placements)
            ]

        if selected_events:
            filtered_df = filtered_df[
                filtered_df["event_name"].isin(selected_events)
            ]

        # Check all six Pokémon slots
        if selected_pokemon:

            pokemon_mask = pd.Series(
                True,
                index=filtered_df.index
            )

            for pokemon in selected_pokemon:
                pokemon_mask &= (
                    filtered_df[pokemon_columns]
                    .eq(pokemon)
                    .any(axis=1)
                )

            filtered_df = filtered_df[pokemon_mask]

        # ------------------------------------------
        # Display Results
        # ------------------------------------------

        st.divider()

        st.subheader("Team Results")

        col1, col2 = st.columns(2)

        col1.metric(
            "Matching Teams",
            len(filtered_df)
        )

        col2.metric(
            "Total Teams",
            len(team_df)
        )

        st.dataframe(
            filtered_df,
            width="stretch",
            hide_index=True
        )