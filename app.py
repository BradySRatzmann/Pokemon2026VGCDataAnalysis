import streamlit as st
import os
import pickle
import shutil

from analysis.team_analysis import run_analysis

from pages.dashboard import render_dashboard
from pages.usage_statistics import render_usage_statistics
from pages.pokemon_explorer import render_pokemon_explorer
from pages.player_analysis import render_player_analysis
from pages.event_analysis import render_event_analysis
from pages.team_predictor import render_team_predictor
from pages.team_database import render_team_database
from pages.team_archetypes import render_team_archetypes
# --------------------------------------------------
# Saved Analysis Settings
# --------------------------------------------------

OUTPUT_DIR = "outputs"
CACHE_FILE = os.path.join(OUTPUT_DIR, "analysis_results.pkl")




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
    page_icon="https://www.smogon.com/forums/media/minisprites/oshawott.png",
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
    width="stretch"
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
        render_dashboard(results)
    # ----------------------------------------------
    # Usage Statistics
    # ----------------------------------------------

    elif page == "Usage Statistics":
        render_usage_statistics(results)

    # Pokémon Explorer
    # ----------------------------------------------

    elif page == "Pokémon Explorer":
        render_pokemon_explorer(results)

    # ----------------------------------------------
    # Player Analysis
    # ----------------------------------------------

    elif page == "Player Analysis":
        render_player_analysis(results)

    # ----------------------------------------------
    # Event Analysis
    # ----------------------------------------------

    elif page == "Event Analysis":
        render_event_analysis(results)

    elif page == "Team Predictor":
        render_team_predictor(results)

    # ----------------------------------------------
    # Team Database
    # ----------------------------------------------

    elif page == "Team Database":
        render_team_database(results)



    # ----------------------------------------------
    # Team Archetypes
    # ----------------------------------------------

    elif page == "Team Archetypes":
        render_team_archetypes(results)

#streamlit run app.py