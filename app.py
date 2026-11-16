
import streamlit as st
import pandas as pd
import os
import pickle
import shutil
import re

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

def get_placement_rank(placement):
    """
    Convert different placement formats into a number
    so players can be sorted from best to worst.
    """
    placement = str(placement).strip().lower()

    # Winning placements
    if placement in ["winner", "winning"]:
        return 1

    # Runner-up placements
    if placement in ["runner-up", "runner up", "runnerup"]:
        return 2

    # Ordinal placements: 1st, 2nd, 3rd, 4th, etc.
    ordinal_match = re.match(r"^(\d+)(st|nd|rd|th)", placement)
    if ordinal_match:
        return int(ordinal_match.group(1))

    # Top placements: Top 4, Top 16, Top 128, etc.
    top_match = re.match(r"^top\s+(\d+)", placement)
    if top_match:
        return int(top_match.group(1))

    # Unrecognized placements go to the bottom of the list
    return 9999
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
    "team composition trends. Currently configured to work with Pokémon Champions team pastes."
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
        "Player Analysis",
        "Event Analysis",
        "Team Predictor",
        "Team Database",
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
                st.image(
                    get_pokemon_image(pokemon["name"]),
                    width=50
                )



    # ----------------------------------------------
    # Usage Statistics
    # ----------------------------------------------

    elif page == "Usage Statistics":

        st.header("Usage Statistics")

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


    # ----------------------------------------------
    # Player Analysis
    # ----------------------------------------------

    elif page == "Player Analysis":

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

                col1.metric(
                    "Total Teams",
                    len(player_teams)
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
                        use_container_width=True
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
                        use_container_width=True
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

    # ----------------------------------------------
    # Event Analysis
    # ----------------------------------------------

    elif page == "Event Analysis":

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
                event for event in team_df["event_name"].dropna().unique()
                if event != "Unknown"
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
                        use_container_width=True
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
                    use_container_width=True,
                    hide_index=True
                )


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
    # Team Database
    # ----------------------------------------------

    elif page == "Team Database":

        st.header("Team Database")

        st.write(
            "Browse the complete team dataset and filter teams "
            "by player, placement, event, or Pokémon."
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

            # Keep only one team per player per event
            filtered_df = team_df.drop_duplicates(
                subset=["player_name", "event_name"],
                keep="first"
            ).copy()

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
                use_container_width=True,
                hide_index=True
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