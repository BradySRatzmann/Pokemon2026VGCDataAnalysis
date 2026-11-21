import re
import requests
import streamlit as st
import random
#Loads Pokemon images into the webapp

def fix_pokemon_names(pokemon):
    replacements = {
        "mega-x": "megax",
        "mega-y": "megay",
        "mega-z": "megaz",
        "-o" : "o"
    }

    for suffix, replacement in replacements.items():
        if pokemon.endswith(suffix):
            pokemon = pokemon[:-len(suffix)] + replacement
            break

    return pokemon

@st.cache_data
def get_pokemon_image(pokemon_name):
    # Convert Pokémon name into the URL format
    pokemon_name = str(pokemon_name).strip().lower().replace(" ", "-")
    pokemon_name = pokemon_name.replace("’", "'").replace("'", "")
    pokemon_name = fix_pokemon_names(pokemon_name)
    return (
        f"https://play.pokemonshowdown.com/sprites/ani/"
        f"{pokemon_name}.gif"
    )
@st.cache_data
def get_pokemon_image_backup(pokemon_name):
    # Convert Pokémon name into the URL format
    pokemon_name = str(pokemon_name).strip().lower().replace(" ", "-")
    pokemon_name = pokemon_name.replace("’", "'").replace("'", "")
    pokemon_name = fix_pokemon_names(pokemon_name)

    return (
        f'https://play.pokemonshowdown.com/sprites/gen5/'
        f"{pokemon_name}.png"
    )

def is_shiny(shiny_rolls=1):
    rolls = 0
    while rolls < shiny_rolls:
        if random.randint(1,4096) == 1:
            return True
        rolls = rolls+1
    return False


@st.cache_data
def get_pokemon_image_shiny(pokemon_name):
    # Convert Pokémon name into the URL format
    pokemon_name = str(pokemon_name).strip().lower().replace(" ", "-")
    pokemon_name = pokemon_name.replace("’", "'").replace("'", "")
    pokemon_name = fix_pokemon_names(pokemon_name)
    return (
        f"https://play.pokemonshowdown.com/sprites/ani-shiny/"
        f"{pokemon_name}.gif"
    )
@st.cache_data
def get_pokemon_image_backup_shiny(pokemon_name):
    # Convert Pokémon name into the URL format
    pokemon_name = str(pokemon_name).strip().lower().replace(" ", "-")
    pokemon_name = pokemon_name.replace("’", "'").replace("'", "")
    pokemon_name = fix_pokemon_names(pokemon_name)

    return (
        f'https://play.pokemonshowdown.com/sprites/gen5-shiny/'
        f"{pokemon_name}.png"
    )

def show_pokemon_image(pokemon_name, width=100):
    height = width
    shiny_roll = is_shiny()
    #I'm so silly goofy XD
    if shiny_roll:
        primary_url = get_pokemon_image_shiny(pokemon_name)
    else:
        primary_url = get_pokemon_image(pokemon_name)

    if shiny_roll:
        backup_url = get_pokemon_image_backup(pokemon_name)
    else:
        backup_url = get_pokemon_image_backup(pokemon_name)

    html = f"""
    <style>
    html, body {{
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
        }}
    </style>
    <div style="
        width: {width}px;
        height: {height}px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        padding: 5px;
        overflow: visible;
    ">
        <img id="pokemon-sprite"
             src={json.dumps(primary_url)}
             style="
                width: 100%;
                height: 100%;
                object-fit: contain;
                object-position: center;
                display: block;
             ">

        <span id="fallback-text"
              style="display:none; color:gray;">
            Sprite unavailable
        </span>
    </div>

    <script>
        const img = document.getElementById("pokemon-sprite");
        const backup = {json.dumps(backup_url)};

        img.addEventListener("error", function() {{
            if (!img.dataset.fallback) {{
                img.dataset.fallback = "true";
                img.src = backup;
            }} else {{
                img.style.display = "none";
                document.getElementById("fallback-text").style.display = "block";
            }}
        }});
    </script>
    """

    st.iframe(
        src="data:text/html;charset=utf-8," + quote(html),
        width=width,
        height=height
    )
    if shiny_roll:
        st.markdown(
            """
            <style>
                .stAudio {
                    display: none;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        #Each element needs somthing unique so I figured the rand int would do it in the rare case a user loads a page with two shinys of the same pokemon species
        st.audio("audio/pla_shiny.mp3", autoplay=True, alt=f"Shiny {pokemon_name}_{random.randint(1,10000000000)}")
