import re
import pandas as pd
from collections import Counter

# Extract player, placement, and event information from a Poképaste title
def parse_team_metadata(title: str):
    pattern = (
        r"^(?P<player>.+?)['’]s\s+"
        r"(?P<placement>Runner[- ]?up|Winner|Winning|Top\s+\d+|"
        r"\d+(?:st|nd|rd|th))\s+"
        r"(?P<event>.+?)\s+Team List$"
    )

    match = re.match(pattern, title.strip(), re.IGNORECASE)

    if not match:
        return {
            "player_name": "Unknown",
            "placement": "Unknown",
            "event_name": "Unknown"
        }

    return {
        "player_name": match.group("player").strip(),
        "placement": match.group("placement").strip(),
        "event_name": re.sub(
    r"\s+Open$",
    "",
    match.group("event").strip(),
    flags=re.IGNORECASE
)
    }

# Function to remove unwanted words from a list of words
def word_removal(words: list[str], remove: list[str]):
    return [word for word in words if word not in remove]

# Function to clean and combine certain words into compound terms
def cleaned_or_Combined(words: list[str], remove: list[str]):
    cleaned = []
    i = 0

    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    while i < len(words):
        if words[i] in remove or is_int(words[i]):
            i += 1
            continue

        combined = [words[i]]
        j = i + 1
        while j < len(words) and words[j] not in remove and not is_int(words[j]):
            combined.append(words[j])
            j += 1

        cleaned.append("-".join(combined))
        i = j
    return cleaned

# Function to tokenize a piece of text formatted for the pokepast.es website
def tokenize(sentence: str):
    lines = [
        line.strip()
        for line in sentence.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    def clean_value(value):
        return value.strip().lower().replace(" ", "-")

    # Default values
    name = "none"
    item = "none"
    ability = "none"
    nature = "none"
    moves = []

    # Read Pokémon name and held item
    header_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        lower_line = line.lower()

        if (
            lower_line.startswith("ability:")
            or lower_line.endswith(" nature")
            or lower_line == "nature"
            or lower_line.startswith("-")
        ):
            break

        header_lines.append(line)
        i += 1

    # Combine header in case HTML extraction split it across lines
    header = " ".join(header_lines)

    if "@" in header:
        name_part, item_part = header.split("@", 1)
        name = clean_value(name_part)
        item = clean_value(item_part)
    else:
        name = clean_value(header)

    # Read ability, nature, and moves
    while i < len(lines):
        line = lines[i].strip()
        lower_line = line.lower()

        # Ability
        if lower_line.startswith("ability:"):
            value = line.split(":", 1)[1].strip()

            if not value and i + 1 < len(lines):
                i += 1
                value = lines[i]

            ability = clean_value(value)

        # Nature
        elif lower_line.endswith(" nature"):
            nature = clean_value(
                line[:-len(" nature")]
            )

        # Standalone Nature label, if HTML splits it
        elif lower_line == "nature" and i > 0:
            nature = clean_value(lines[i - 1])

        # Moves
        elif line == "-":
            if i + 1 < len(lines):
                i += 1
                moves.append(clean_value(lines[i]))

        elif line.startswith("-"):
            moves.append(clean_value(line[1:]))

        i += 1

    # Keep exactly four move slots
    moves = (moves + ["none"] * 4)[:4]

    return [name, item, ability, nature] + moves

# Convert placements into numeric rankings
def get_placement_rank(placement: str):
    placement = str(placement).strip().lower()

    if placement in ["winner", "winning"]:
        return 1

    if placement in ["runner-up", "runner up", "runnerup"]:
        return 2

    # Handle placements like Top 8, Top 64, Top 256
    top_match = re.match(r"top\s+(\d+)", placement)

    if top_match:
        return int(top_match.group(1))

    # Handle placements like 1st, 2nd, 3rd, 256th
    ordinal_match = re.match(r"(\d+)(?:st|nd|rd|th)", placement)

    if ordinal_match:
        return int(ordinal_match.group(1))

    # Unknown or unrecognized placements
    return 9999





# Build DataFrames based on tokenized data
# -----------------------------------------------------------------------------


# Formatting for Pokémon DataFrame rows
def data_to_row(tokens: list[str]):
    columns = [
        "name",
        "item",
        "ability",
        "nature",
        "move_1",
        "move_2",
        "move_3",
        "move_4",
    ]

    # Pad or truncate safely
    fixed = (tokens + ["none"] * 8)[:8]

    return dict(zip(columns, fixed))




# Build team-level DataFrame with one row per team.
# Each team gets a unique ID, the URL, and the six Pokémon names.
# Combines the original URL with its tokenized Pokémon data and metadata
def build_team_tokenized(data_points, paragraphs_list):
    teams = []

    for url, page in zip(data_points, paragraphs_list):
        pokemons = [
            tokenize(p) for p in page["paragraphs"]
        ]

        metadata = page["metadata"]

        teams.append((url, pokemons, metadata))

    return teams


# Build team-level DataFrame with one row per team.
# Each team gets a unique ID, URL, metadata, and six Pokémon names.
def build_team_dataframe(team_tokenized):
    rows = []

    for team_id, (url, pokemons, metadata) in enumerate(
        team_tokenized, start=1
    ):
        row = {
            "team_id": team_id,
            "source_url": url,
            "player_name": metadata["player_name"],
            "placement": metadata["placement"],
            "event_name": metadata["event_name"],
        }

        for i, pkm in enumerate(pokemons, start=1):
            name = pkm[0] if pkm else None
            row[f"pokemon_{i}"] = name

        rows.append(row)

    return pd.DataFrame(rows)
