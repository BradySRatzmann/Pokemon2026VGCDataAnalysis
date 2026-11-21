def render_team_archetypes(results):
    st.header("Team Archetypes")

    st.write(
        "Explore the most common team compositions identified "
        "by the machine learning clustering algorithm."
    )

    archetype_df = results["archetype_df"]
    team_clusters = results["team_clusters"]

    st.caption(
        f"Clustering identified {results['best_k']} archetypes "
        f"using a silhouette score of {results['best_score']:.3f}."
    )

    if archetype_df.empty:
        st.warning("No team archetypes were found.")

    else:
        total_teams = len(team_clusters)

        archetype_df = archetype_df.copy()

        archetype_df["percentage"] = (
                archetype_df["num_teams"] / total_teams * 100
        )

        # Show the 10 largest archetypes
        top_archetypes = archetype_df.head(10)

        st.subheader("Top 10 Most Common Team Archetypes")

        for rank, (_, archetype) in enumerate(
                top_archetypes.iterrows(),
                start=1
        ):

            with st.expander(
                    f"#{rank} - {archetype['archetype_name']} "
                    f"({int(archetype['num_teams'])} teams)"
            ):

                col1, col2 = st.columns(2)

                col1.metric(
                    "Teams",
                    int(archetype["num_teams"])
                )

                col2.metric(
                    "Dataset Share",
                    f"{archetype['percentage']:.1f}%"
                )

                st.write("**Core Pokémon**")

                core_pokemon = [
                    pokemon.strip()
                    for pokemon in archetype["core_pokemon"].split(";")
                    if pokemon.strip()
                ]

                if core_pokemon:
                    sprite_columns = st.columns(len(core_pokemon))

                    for column, pokemon in zip(
                            sprite_columns,
                            core_pokemon
                    ):
                        with column:
                            show_pokemon_image(pokemon, width=90)
                            st.caption(pokemon.title())

                st.divider()

                st.write("**Most Common Pokémon in This Archetype**")

                top_pokemon = [
                    pokemon.split("(")[0].strip()
                    for pokemon in archetype["top_mons"].split(";")
                    if pokemon.strip()
                ]

                sprite_columns = st.columns(5)

                for i, pokemon in enumerate(top_pokemon[:10]):
                    with sprite_columns[i % 5]:
                        show_pokemon_image(pokemon, width=75)
                        st.caption(pokemon.title())

                st.caption(
                    "Core Pokémon are selected based on how "
                    "distinctive they are to this archetype. "
                    "The remaining Pokémon are ranked by frequency."
                )

        st.divider()

        st.subheader("Archetype Distribution")

        chart_df = top_archetypes.copy()

        chart_df["Archetype"] = chart_df["archetype_name"]

        st.bar_chart(
            chart_df.set_index("Archetype")["num_teams"]
        )

        st.subheader("Archetype Data")

        st.dataframe(
            archetype_df[
                [
                    "archetype_name",
                    "num_teams",
                    "percentage",
                    "core_pokemon",
                    "top_mons"
                ]
            ].rename(columns={
                "archetype_name": "Archetype",
                "num_teams": "Teams",
                "percentage": "Dataset Share (%)",
                "core_pokemon": "Core Pokémon",
                "top_mons": "Most Common Pokémon"
            }),
            width="stretch",
            hide_index=True
        )