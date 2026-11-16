
import streamlit as st
import pandas as pd
import os
import pickle
import shutil

from main import (
    run_analysis,
    analyze_pokemon_partners
)


# --------------------------------------------------
# Saved Analysis Settings
# --------------------------------------------------

OUTPUT_DIR = "outputs"
CACHE_FILE = os.path.join(OUTPUT_DIR, "analysis_results.pkl")

def get_pokemon_image(pokemon_name):
    # Convert Pokémon name into the URL format
    pokemon_name = str(pokemon_name).strip().lower().replace(" ", "-")
    pokemon_name = pokemon_name.replace("’", "'").replace("'", "")

    return (
        f"https://www.smogon.com/forums/media/minisprites/"
        f"{pokemon_name}.png"
    )


def show_pokemon_image(pokemon_name, width=100):
    # Keep the page usable if a sprite URL is unavailable.
    try:
        st.image(get_pokemon_image(pokemon_name), width=width)
    except Exception:
        st.caption("Sprite unavailable")
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Pokémon VGC Analyzer",
    page_icon="⚔️",
    layout="wide"
)


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title("⚔️ Pokémon VGC Analyzer")

st.write(
    "Explore competitive Pokémon teams, "
    "analyze usage statistics, and discover "
    "team composition trends. Currently Configured to work with Pokemon Scarlet/Violet OTS poke-pastes"
)

st.divider()


# --------------------------------------------------
# Sidebar - Data Input
# --------------------------------------------------

st.sidebar.title("Data Settings")

data_source = st.sidebar.radio(
    "Choose your data source:",
    ["Existing data.txt", "Upload a URL list"]
)

uploaded_file = None

if data_source == "Upload a URL list":

    uploaded_file = st.sidebar.file_uploader(
        "Upload a text file",
        type=["txt"]
    )

else:
    st.sidebar.caption(
        "Using the URLs stored in data.txt"
    )



# --------------------------------------------------
# Run Analysis
# --------------------------------------------------

if st.sidebar.button(
    "Run Analysis",
    type="primary",
    use_container_width=True
):

    data_points = None
    should_run = False

    if data_source == "Existing data.txt":

        if os.path.exists("data.txt"):
            should_run = True
        else:
            st.sidebar.error("Could not find data.txt.")

    else:

        if uploaded_file is None:
            st.sidebar.error("Please upload a text file.")

        else:
            contents = uploaded_file.getvalue().decode("utf-8")

            data_points = [
                line.strip()
                for line in contents.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]

            if data_points:
                should_run = True
            else:
                st.sidebar.error("No URLs found in the file.")

    if should_run:

        with st.spinner(
                "Fetching and analyzing Pokémon teams..."
        ):

            # Remove previous generated analysis files
            if os.path.exists(OUTPUT_DIR):
                shutil.rmtree(OUTPUT_DIR)

            # Run a fresh analysis
            results = run_analysis(data_points)

            # Save results so they can be reused next time
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            with open(CACHE_FILE, "wb") as file:
                pickle.dump(results, file)

            st.session_state["results"] = results

            st.sidebar.success("Analysis completed!")



# --------------------------------------------------
# Navigation
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Usage Statistics",
        "Team Synergy",
        "Pokémon Explorer",
        "Team Predictor",
        "Team Archetypes"
    ]
)

# --------------------------------------------------
# Load Previously Saved Results
# --------------------------------------------------

if (
    "results" not in st.session_state
    and os.path.exists(CACHE_FILE)
):

    try:
        with open(CACHE_FILE, "rb") as file:
            st.session_state["results"] = pickle.load(file)

        st.sidebar.success("Loaded previously saved analysis!")

    except Exception as e:
        st.sidebar.warning(
            f"Could not load saved analysis: {e}"
        )

# --------------------------------------------------
# Display Results
# --------------------------------------------------

if "results" not in st.session_state:

    st.info(
        "Choose a data source and click Run Analysis "
        "to get started."
    )

else:

    results = st.session_state["results"]

    pokemon_df = results["pokemon_df"]
    team_df = results["team_df"]
    usage_results = results["usage_results"]

    # ----------------------------------------------
    # Dashboard
    # ----------------------------------------------

    if page == "Dashboard":

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

        st.bar_chart(
            usage_df,
            x="name",
            y="usage_rate"
        )

        st.subheader("Team Data Preview")

        st.dataframe(
            team_df.head(10),
            use_container_width=True
        )

    # ----------------------------------------------
    # Usage Statistics
    # ----------------------------------------------

    elif page == "Usage Statistics":

        st.header("Usage Statistics")

        usage_mapping = {

            "Pokémon": "pokemon_usage",

            "Items": "item_usage",

            "Tera Types": "tera_usage",

            "Moves": "move_usage"

        }

        analysis_type = st.selectbox(

            "Select a category:",

            list(usage_mapping.keys())

        )

        selected_data = results["usage_results"][

            usage_mapping[analysis_type]

        ]

        # Moves use total_uses instead of usage_rate

        value_column = (

            "total_uses" if analysis_type == "Moves"

            else "usage_rate"

        )

        st.subheader(f"{analysis_type} Usage")

        st.bar_chart(selected_data[value_column])

        st.dataframe(selected_data)

# --------------------------------------------------
# Team Synergy
# --------------------------------------------------

    elif page == "Team Synergy":

        st.header("Team Synergy")

        st.write(
        "Explore which Pokémon commonly appear together "
        "on competitive teams."
     )

        # Get pair usage data
        pair_df = results["pair_df"]

        if pair_df.empty:
            st.warning("No Pokémon pair data was found.")

        else:

            # ------------------------------------------
            # Most Common Pokémon Pairs
            # ------------------------------------------

            st.subheader("Most Common Pokémon Pairs")

            top_pairs = pair_df.head(20).copy()

            # Make Pokémon pairs easier to read
            top_pairs.index = [
                f"{a} + {b}"
                for a, b in top_pairs.index
            ]

            st.bar_chart(top_pairs["teams_used"])

            st.caption(
            "   Shows how many teams used each Pokémon combination."
         )

            # Display pair data
            st.dataframe(
                top_pairs,
                use_container_width=True
            )

            # ------------------------------------------
            # Pokémon Partner Explorer
            # ------------------------------------------

            st.divider()

            st.subheader("Pokémon Partner Explorer")

            pokemon_options = sorted(
                results["pokemon_df"]["name"]
                .dropna()
                .unique()
            )

            selected_pokemon = st.selectbox(
                "Choose a Pokémon:",
                pokemon_options
            )
            # Display selected Pokémon
            col1, col2 = st.columns([1, 4])

            with col1:
                show_pokemon_image(selected_pokemon, width=100)

            with col2:
                st.subheader(selected_pokemon.title())

            partners_df = analyze_pokemon_partners(
                pair_df,
                selected_pokemon,
                top_n=10
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
                        show_pokemon_image(partner["partner"], width=80)

                        st.caption(partner["partner"].title())

                        st.caption(
                            f'{int(partner["teams_used"])} teams'
                        )

                # Existing partner chart
                st.bar_chart(
                    partners_df.set_index("partner")["teams_used"]
                )

                # Existing partner table
                st.dataframe(
                    partners_df,
                    use_container_width=True
                )





    # ----------------------------------------------
    # Pokémon Explorer
    # ----------------------------------------------

    elif page == "Pokémon Explorer":

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
                show_pokemon_image(selected_pokemon, width=120)

            pokemon_usage = usage_results.get(
                "pokemon_usage", pd.DataFrame()
            )

            with info_col:
                st.subheader(selected_pokemon.title())

                if selected_pokemon in pokemon_usage.index:
                    usage_row = pokemon_usage.loc[selected_pokemon]
                    metric_col1, metric_col2 = st.columns(2)

                    metric_col1.metric(
                        "Teams Used",
                        int(usage_row["teams_used"])
                    )

                    metric_col2.metric(
                        "Usage Rate",
                        f'{usage_row["usage_rate"]:.1f}%'
                    )

                    pokemon_rank = (
                        pokemon_usage.index.get_loc(selected_pokemon) + 1
                    )
                    st.caption(
                        f"Usage ranking: #{pokemon_rank} "
                        f"out of {len(pokemon_usage)} Pokémon"
                    )

            st.divider()

            # Common attributes are calculated from the already-loaded
            # Pokémon rows; no additional scraping or analysis run is needed.
            attribute_columns = st.columns(3)

            attribute_sections = [
                ("Most Common Items", "item"),
                ("Most Common Abilities", "ability"),
                ("Most Common Tera Types", "tera_type")
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
                        st.dataframe(
                            counts.rename("Uses").to_frame(),
                            use_container_width=True
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
                        use_container_width=True
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
                    st.dataframe(
                        matching_teams[["team_id"] + team_columns],
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.caption("Team membership data is not available.")




    elif page == "Team Predictor":
        st.title("Team Predictor")
        st.write(
            "Select two or more Pokémon to discover which other "
            "Pokémon commonly appear alongside them on successful teams."
        )

        # Get the Pokémon columns from the team dataframe
        pokemon_columns = [
            f"pokemon_{i}" for i in range(1, 7)
            if f"pokemon_{i}" in team_df.columns
        ]

        if not pokemon_columns:
            st.warning("Could not find Pokémon columns in the team data.")

        else:
            # Get all available Pokémon
            all_pokemon = sorted(
        set(
            str(pokemon).strip()
            for pokemon in team_df[pokemon_columns].values.flatten()
            if pd.notna(pokemon) and str(pokemon).strip() != ""
            )
        )

            # Let the user select multiple Pokémon
            # Let the user select multiple Pokémon
            selected_pokemon = st.multiselect(
                "Select at least two Pokémon:",
                all_pokemon,
                placeholder="Choose Pokémon..."
            )

            # Display selected Pokémon sprites
            if selected_pokemon:
                st.subheader("Selected Pokémon")

                sprite_columns = st.columns(len(selected_pokemon))

                for column, pokemon in zip(sprite_columns, selected_pokemon):
                    with column:
                        show_pokemon_image(pokemon, width=80)
                        st.caption(pokemon)

            # Number of recommendations
            top_n = st.slider(
                "Number of recommendations:",
                min_value=5,
                max_value=20,
                value=10
            )

            if len(selected_pokemon) < 2:
                st.info("Select at least two Pokémon to generate predictions.")

            else:
                # Find teams containing ALL selected Pokémon
                matching_teams = []

                for _, team in team_df.iterrows():
                    team_members = set(
                        str(team[col])
                        for col in pokemon_columns
                        if pd.notna(team[col])
                    )

                    if set(selected_pokemon).issubset(team_members):
                        matching_teams.append(team_members)

                st.subheader("Prediction Results")

                if not matching_teams:
                    st.warning(
                        "No teams were found containing all selected Pokémon."
                    )

                else:
                    # Count how often each other Pokémon appears
                    partner_counts = {}

                    for team in matching_teams:
                        for pokemon in team:
                            if pokemon not in selected_pokemon:
                                partner_counts[pokemon] = (
                                        partner_counts.get(pokemon, 0) + 1
                                )

                    # Convert results into a dataframe
                    prediction_df = pd.DataFrame(
                        [
                            {
                                "Pokémon": pokemon,
                                "Teams Used": count,
                                "Usage Rate": count / len(matching_teams) * 100
                            }
                            for pokemon, count in partner_counts.items()
                        ]
                    )

                    if prediction_df.empty:
                        st.info("No additional Pokémon were found.")
                    else:
                        prediction_df = prediction_df.sort_values(
                            "Teams Used",
                            ascending=False
                        ).head(top_n)

                        st.write(
                            f"Found **{len(matching_teams)} teams** "
                            f"containing all selected Pokémon."
                        )

                        # Display recommendations
                        for _, row in prediction_df.iterrows():
                            col1, col2 = st.columns([1, 4])

                            with col1:
                                show_pokemon_image(row["Pokémon"], width=80)

                            with col2:
                                st.markdown(f"**{row['Pokémon']}**")
                                st.write(
                                    f"Appeared on {int(row['Teams Used'])} "
                                    f"matching teams "
                                    f"({row['Usage Rate']:.1f}%)"
                                )

                        # Show the complete results in a table
                        st.subheader("Prediction Data")
                        st.dataframe(
                            prediction_df,
                            hide_index=True,
                            use_container_width=True
                        )
    # ----------------------------------------------
    # Team Archetypes
    # ----------------------------------------------

    elif page == "Team Archetypes":

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
                use_container_width=True,
                hide_index=True
            )
#streamlit run app.py