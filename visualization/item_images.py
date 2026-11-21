import requests
import streamlit as st
@st.cache_data
def get_item_image(item_name):
    # Convert item name into the Pokémon Showdown URL format
    item_name = str(item_name).strip().lower()
    item_name = item_name.replace(" ","-")

    image_url = (
        f"https://play.pokemonshowdown.com/sprites/itemicons/"
        f"{item_name}.png"
    )

    backup_url = (
        "https://play.pokemonshowdown.com/sprites/itemicons/0.png"
    )

    mega_stones = ["ite","ite-x","ite-y","ite-z"]

    for suffix in mega_stones:
        if item_name.endswith(suffix):
            image_url = "https://play.pokemonshowdown.com/sprites/misc/mega.png"

    # Check whether the item image exists
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        return image_url

    except requests.RequestException:
        return backup_url


def show_item_image(item_name, width=40):
    image_url = get_item_image(item_name)
    backup_url = "https://play.pokemonshowdown.com/sprites/itemicons/0.png"

    html = f"""
    <div style="
        width: {width}px;
        height: {width}px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
        padding: 3px;
    ">
        <img id="item-sprite"
             src={json.dumps(image_url)}
             style="
                width: 100%;
                height: 100%;
                object-fit: contain;
                object-position: center;
                display: block;
             ">
    </div>

    <script>
        const img = document.getElementById("item-sprite");
        const backup = {json.dumps(backup_url)};

        img.addEventListener("error", function() {{
            if (!img.dataset.fallback) {{
                img.dataset.fallback = "true";
                img.src = backup;
            }}
        }});
    </script>
    """

    st.iframe(
        src="data:text/html;charset=utf-8," + quote(html),
        width=width,
        height=width
    )