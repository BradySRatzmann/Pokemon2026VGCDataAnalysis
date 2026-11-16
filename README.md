# Pokémon VGC Data Analysis Project

## Overview

This project analyzes competitive Pokémon VGC team data from the **2027 VGC season (2026–2027)** to compute Pokémon usage statistics, team synergy, partner frequency, player performance, event results, and other analytical metrics.

The project extracts competitive teams from Poképaste links, organizes the information into structured datasets, performs statistical and machine learning analysis, and generates visualizations.

To keep the dataset focused on higher-performing competitive teams, the analysis pipeline only includes teams with a recorded placement of **Top 256 or better**. Teams with unknown or unrecognized placements are excluded.

The project also includes an interactive Streamlit application that allows users to explore Pokémon statistics, examine team compositions, compare player and event performance, and predict potential team members based on selected Pokémon.

## Project Structure

### Development Version

```text
project_root/
│
├── app.py                              # Streamlit user interface
├── main.py                             # Main data collection and analysis pipeline
├── launcher.py                         # Launcher for packaged Windows application
│
├── data.txt                            # Main dataset input
├── testData.txt                        # Optional smaller test dataset
│
├── outputs/                            # Automatically generated analysis files
│   ├── attributes/                      # Pokémon attribute breakdowns
│   ├── data/                            # Processed Pokémon and team datasets
│   ├── partners/                        # Pokémon partner breakdowns
│   ├── synergy/                         # Pair usage and synergy statistics
│   ├── ml/                              # Machine learning and team archetype clustering
│   └── usage/                           # Individual Pokémon usage statistics
│
├── requirements.txt                    # Python dependencies
└── README.md                            # Project documentation
```

## Streamlit Application

The project includes an interactive web interface built using Streamlit.

The application currently contains the following pages:

- **Dashboard:** Provides an overview of the collected team data, including commonly used Pokémon and general usage statistics.

- **Usage Statistics:** Displays Pokémon usage information, including usage rates and related statistics.

- **Pokémon Explorer:** Allows users to select an individual Pokémon and explore its usage statistics, most common items, abilities, natures, moves, and partner Pokémon. It also displays the Pokémon's highest recorded placement, the player and event associated with that placement, and the teams it appears on.

- **Player Analysis:** Allows users to examine individual players, including their recorded teams, most-used Pokémon, and most common placements.

- **Event Analysis:** Provides an overview of individual tournaments, including the most commonly used Pokémon and the top 16 recorded players with their placements and teams.

- **Team Predictor:** Allows users to select two or more Pokémon and find other Pokémon that commonly appear alongside all selected Pokémon.

- **Team Database:** Provides a searchable and filterable collection of teams. Users can filter by player, placement, event, and Pokémon.

- **Team Archetypes:** Uses machine learning and clustering to group teams based on their six Pokémon, helping identify common team compositions and archetypes.

The application also incorporates Pokémon sprites throughout the interface where appropriate to make the results easier to explore visually.

### Pokémon Explorer

The Pokémon Explorer provides detailed information about an individual Pokémon using the existing team dataset.

Users can select a Pokémon to view:

- Total number of teams featuring the Pokémon.
- Overall usage rate and usage ranking.
- Most common items, abilities, natures, and moves.
- Most common Pokémon partners, including the number of teams they appear together on.
- The Pokémon's highest recorded tournament placement.
- The player and event associated with its highest placement.
- A list of teams featuring the selected Pokémon, including player names, placements, events, and team compositions.

The Partner Explorer is integrated directly into this page and uses the same Pokémon selection as the rest of the Pokémon Explorer.

### Player Analysis

The Player Analysis page provides an overview of individual competitors in the dataset.

Users can select a player to examine:

- Their recorded teams.
- Their most frequently used Pokémon.
- Their most common tournament placements.
- The events in which they have recorded teams.

### Event Analysis

The Event Analysis page provides a breakdown of individual tournaments.

For a selected event, the application displays:

- Total number of recorded teams.
- Number of participating players.
- Number of unique Pokémon used.
- Most commonly used Pokémon.
- A leaderboard of the top 16 recorded players, including their placements and team compositions.

Each player is displayed only once per event.

### Team Predictor

The Team Predictor identifies potential Pokémon partners based on existing team data.

Users select at least two Pokémon, and the application searches for teams containing **all selected Pokémon**. It then counts how frequently other Pokémon appear within those matching teams.

Recommendations are ranked by how many matching teams contain each Pokémon. The application also displays the percentage of matching teams in which each recommended Pokémon appears.

### Team Database

The Team Database provides access to the collected team records in a searchable interface.

Users can filter the database using:

- Player name
- Tournament placement
- Event name
- Pokémon appearing on the team

When multiple Pokémon are selected, the database only displays teams containing all selected Pokémon.

### Team Archetypes

The Team Archetypes page uses Scikit-Learn to analyze team compositions and identify groups of similar teams.

The clustering process uses the presence of Pokémon on each team rather than individual attributes such as items, moves, or abilities.

Teams are converted into Pokémon-presence vectors using `MultiLabelBinarizer`, reduced using `TruncatedSVD`, and grouped using `KMeans`.

Cluster quality is evaluated using silhouette scores to help determine an appropriate number of archetypes.

The resulting clusters are summarized using their most distinctive and frequently appearing Pokémon.

## Requirements

To run the development version, install the following Python packages:

```bash
pip install streamlit aiohttp beautifulsoup4 lxml pandas numpy matplotlib scikit-learn
```

For building the Windows executable, PyInstaller is also required:

```bash
pip install pyinstaller
```

## Running the Application

### Running the Streamlit Interface

From the project directory, run:

```bash
streamlit run app.py
```

The application will open in your default web browser. If it does not open automatically, navigate to:

```text
http://localhost:8501
```

### Running the Analysis Pipeline Directly

The analysis pipeline can also be run independently using:

```bash
python main.py
```

The main dataset is configured to use `data.txt` by default. To use a different dataset, change the filename passed to the `read_data()` function in `main.py`.

## Input / Output

### Input

The primary input is `data.txt`, which contains the Poképaste links used by the project.

An optional smaller dataset, `testData.txt`, can be used for testing.

The dataset contains teams from the 2027 VGC season (2026–2027).

### Team Placement Filtering

During analysis, teams are filtered according to their recorded tournament placement.

Only teams that placed **Top 256 or better** are included in the generated datasets and analysis.

Examples of included placements:

- Winner
- Runner-up
- Top 8
- Top 16
- Top 64
- Top 128
- Top 256

Teams with placements worse than Top 256, as well as teams with unknown or unrecognized placements, are excluded.

This filtering occurs before the Pokémon and team DataFrames are constructed. As a result, usage statistics, synergy calculations, partner analysis, player analysis, event analysis, and team archetypes all use the filtered dataset.

### Output

The program generates basic statistics in the terminal along with additional CSV files and visualizations.

Generated files are organized into the following directories:

```text
outputs/
│
├── attributes/                         # Pokémon attribute breakdowns
├── data/                               # Processed Pokémon and team datasets
├── partners/                            # Top Pokémon partner breakdowns
├── synergy/                             # Pair usage and synergy statistics
├── ml/                                  # Team archetype clustering results
└── usage/                               # Individual Pokémon usage statistics
```

The exact output files depend on which analyses are performed.

## Windows Release

The project can be distributed as a standalone Windows application using PyInstaller.

The packaged version runs the Streamlit application locally and automatically opens it in the user's default web browser.

### Release Folder Structure

The Windows release should contain the following:

```text
PokemonVGC/
│
├── PokemonVGC.exe                     # Main application executable
├── data.txt                           # External copy of the main dataset
│
└── _internal/                          # Bundled Python runtime and dependencies
```

**Important:** The entire `PokemonVGC` folder must remain together. The executable depends on the `_internal` directory and cannot run independently.

### Installing and Running the Release

1. Download the Windows ZIP file from the GitHub Releases page.
2. Extract the ZIP file.
3. Open the extracted `PokemonVGC` folder.
4. Double-click `PokemonVGC.exe`.
5. The application will start and automatically open in your default web browser.

If the browser does not open automatically, navigate to:

```text
http://localhost:8501
```

Keep the console window open while using the application. Closing the console window will stop the application.

Python does not need to be installed separately to run the packaged release.

### Data File Export

The packaged application includes `data.txt` as a bundled resource.

When the executable starts, the launcher automatically copies `data.txt` into the same folder as `PokemonVGC.exe` if an external copy does not already exist.

This allows users to access and modify the dataset without needing to access the bundled application files.

An existing external `data.txt` will not be overwritten when the application starts.

### Building the Windows Executable

To build the Windows release, run the following command from the project directory:

```bash
pyinstaller --clean --noconfirm --onedir --console --name PokemonVGC --collect-all streamlit --hidden-import main --add-data "app.py;." --add-data "main.py;." --add-data "data.txt;." launcher.py
```

The packaged application will be generated in:

```text
dist/PokemonVGC/
```

For distribution, compress the entire `PokemonVGC` folder into a ZIP file.

## Datasets

The data used in this project comes from competitive Pokémon VGC teams for the **2027 season (2026–2027)**.

Tournament information and team data are compiled by Nimbasa City Post. The project uses Poképaste links collected from tournament results to build its team database.

The 2027 season information can be found at:

https://www.nimbasacitypost.com/2026/08/pokemon-vgc-2027-season-information.html

An internet connection is required for retrieving online Poképaste data and loading Pokémon sprite images.