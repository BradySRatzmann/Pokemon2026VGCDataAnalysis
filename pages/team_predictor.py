import streamlit as st
import pandas as pd

from analysis.prediction import recommend_sixth_mon
from visualization.pokemon_images import show_pokemon_image

def render_team_predictor(results):
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

            # Retrieve the trained model from the analysis results

            clf = results.get("sixth_mon_model")

            mlb = results.get("sixth_mon_mlb")

            metrics = results.get("sixth_mon_metrics", {})

            if clf is None or mlb is None:

                st.warning(

                    "The prediction model is not available. "

                    "Please run a fresh analysis to train the model."

                )


            else:

                st.subheader("Prediction Results")

                # Generate predictions using the trained model

                recommendations = recommend_sixth_mon(

                    clf,

                    mlb,

                    selected_pokemon,

                    top_n=top_n

                )

                if not recommendations:

                    st.info("No recommendations could be generated.")


                else:

                    st.caption(

                        "Predictions generated using your trained "

                        "Logistic Regression model."

                    )

                    # Display model evaluation metrics

                    if metrics:
                        st.write(

                            f"Model Top-1 Accuracy: "

                            f"{metrics['top1']:.1%}"

                        )

                        st.write(

                            f"Model Top-3 Accuracy: "

                            f"{metrics['top3']:.1%}"

                        )

                    prediction_df = pd.DataFrame(

                        [

                            {

                                "Pokémon": pokemon,

                                "Predicted Probability": probability * 100

                            }

                            for pokemon, probability in recommendations

                        ]

                    )

                    # Display recommendations

                    for _, row in prediction_df.iterrows():
                        col1, col2 = st.columns([1, 4])

                        with col1:
                            show_pokemon_image(row["Pokémon"], width=80)

                        with col2:
                            st.markdown(f"**{row['Pokémon']}**")

                            st.write(

                                f"Predicted Probability: "

                                f"{row['Predicted Probability']:.2f}%"

                            )

                    # Show the complete results in a table

                    st.subheader("Prediction Data")

                    st.dataframe(

                        prediction_df,

                        hide_index=True,

                        width="stretch"

                    )

