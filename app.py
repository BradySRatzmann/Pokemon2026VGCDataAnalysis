
import streamlit as st
import pandas as pd
import os

from main import (
    run_analysis,
    analyze_pokemon_partners
)

def get_pokemon_image(pokemon_name):
    # Convert Pokémon name into the URL format
    pokemon_name = pokemon_name.lower().replace(" ", "-")

    return (
        f"https://www.smogon.com/forums/media/minisprites/"
        f"{pokemon_name}.png"
    )
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
    "team composition trends."
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

        try:
            with st.spinner(
                "Fetching and analyzing Pokémon teams..."
            ):

                results = run_analysis(data_points)

                st.session_state["results"] = results

            st.sidebar.success("Analysis completed!")

        except Exception as e:
            st.error(f"Analysis failed: {e}")


# --------------------------------------------------
# Navigation
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Usage Statistics", "Team Synergy"]
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
                st.image(
                    get_pokemon_image(selected_pokemon),
                    width=100
                )

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
                        st.image(
                            get_pokemon_image(partner["partner"]),
                            width=80
                        )

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



#streamlit run app.py