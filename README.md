\# Pokémon VGC Data Analysis Project



\## Overview



This project analyzes competitive Pokémon VGC team data to compute usage statistics, synergy scores, partner frequency, and other analytical metrics. It supports extracting teams from Poképaste links, organizing the information into structured datasets, running statistical analysis, and generating visualizations.



The project is designed to help players, analysts, and researchers understand which Pokémon combinations appear most often and how frequently they are used in competitive play.



The project also includes an interactive Streamlit application that allows users to explore Pokémon statistics, examine team compositions, and predict potential team members based on selected Pokémon.



\## Project Structure



\### Development Version



```text

project\_root/

│

├── app.py                              # Streamlit user interface

├── main.py                             # Main data collection and analysis pipeline

├── launcher.py                         # Launcher for packaged Windows application

│

├── data.txt                            # Main dataset input

├── testData.txt                        # Optional smaller test dataset

│

├── outputs/                            # Automatically generated analysis files

│   ├── attributes/                      # Top attribute breakdowns (CSVs + plots)

│   ├── data/                            # Processed team data

│   ├── partners/                        # Top Pokémon partner breakdowns

│   ├── synergy/                         # Pair usage and partner statistics

│   ├── ml/                              # Machine learning and team archetype clustering

│   └── usage/                           # Individual Pokémon usage statistics

│

├── requirements.txt                    # Python dependencies

└── README.md                            # Project documentation

```



\## Streamlit Application



The project includes an interactive web interface built using Streamlit.



The application currently contains the following pages:



\- \*\*Dashboard:\*\* Provides an overview of the collected team data and analysis.

\- \*\*Usage Statistics:\*\* Displays Pokémon usage information and related statistics.

\- \*\*Team Synergy:\*\* Explores Pokémon pairings and how frequently Pokémon appear together.

\- \*\*Pokémon Explorer:\*\* Allows users to select an individual Pokémon and explore its usage statistics, attributes, moves, and the teams in which it appears.

\- \*\*Team Predictor:\*\* Allows users to select two or more Pokémon and find other Pokémon that commonly appear alongside them on teams containing all selected Pokémon.



The application also incorporates Pokémon sprites throughout the interface where appropriate to make the results easier to explore visually.



\### Team Predictor



The Team Predictor identifies potential Pokémon partners based on existing team data.



Users select at least two Pokémon, and the application searches for teams containing all selected Pokémon. It then counts how frequently other Pokémon appear within those matching teams.



Recommendations are ranked by how many matching teams contain each Pokémon. The application also displays the percentage of matching teams in which each recommended Pokémon appears.



\## Requirements



To run the development version, install the following Python packages:



```bash

pip install streamlit aiohttp beautifulsoup4 lxml pandas numpy matplotlib scikit-learn

```



For building the Windows executable, PyInstaller is also required:



```bash

pip install pyinstaller

```



\## Running the Application



\### Running the Streamlit Interface



From the project directory, run:



```bash

streamlit run app.py

```



The application will open in your default web browser. If it does not open automatically, navigate to:



```text

http://localhost:8501

```



\### Running the Analysis Pipeline Directly



The analysis pipeline can also be run independently using:



```bash

python main.py

```



The main dataset is configured to use `data.txt` by default. To use a different dataset, change the filename passed to the `read\_data()` function in `main.py`.



\## Input / Output



\### Input



The primary input is `data.txt`, which contains the team data or Poképaste links used by the project.



An optional smaller dataset, `testData.txt`, can be used for testing.



\### Output



The program generates basic statistics in the terminal along with additional CSV files and visualizations.



Generated files are organized into the following directories:



```text

outputs/

│

├── attributes/                         # Top attribute breakdowns

├── data/                               # Processed team data

├── partners/                           # Top partner breakdowns

├── synergy/                            # Pair usage and partner statistics

├── ml/                                 # Team archetype clustering

└── usage/                              # Individual Pokémon usage statistics

```



The exact output files depend on which analyses are performed.



\## Windows Release



The project can be distributed as a standalone Windows application using PyInstaller.



The packaged version runs the Streamlit application locally and automatically opens it in the user's default web browser.



\### Release Folder Structure



The Windows release should contain the following:



```text

PokemonVGC/

│

├── PokemonVGC.exe                     # Main application executable

├── data.txt                           # External copy of the main dataset

│

└── \_internal/                         # Bundled Python runtime and dependencies

```



\*\*Important:\*\* The entire `PokemonVGC` folder must remain together. The executable depends on the `\_internal` directory and cannot run independently.



\### Installing and Running the Release



1\. Download the Windows ZIP file from the GitHub Releases page.

2\. Extract the ZIP file.

3\. Open the extracted `PokemonVGC` folder.

4\. Double-click `PokemonVGC.exe`.

5\. The application will start and automatically open in your default web browser.



If the browser does not open automatically, navigate to:



```text

http://localhost:8501

```



Keep the console window open while using the application. Closing the console window will stop the application.



Python does not need to be installed separately to run the packaged release.



\### Data File Export



The packaged application includes `data.txt` as a bundled resource.



When the executable starts, the launcher automatically copies `data.txt` into the same folder as `PokemonVGC.exe` if an external copy does not already exist.



This allows users to access and modify the dataset without needing to access the bundled application files.



An existing external `data.txt` will not be overwritten when the application starts.



\### Building the Windows Executable



To build the Windows release, run the following command from the project directory:



```bash

pyinstaller --clean --noconfirm --onedir --console --name PokemonVGC --collect-all streamlit --hidden-import main --add-data "app.py;." --add-data "main.py;." --add-data "data.txt;." launcher.py

```



The packaged application will be generated in:



```text

dist/PokemonVGC/

```



For distribution, compress the entire `PokemonVGC` folder into a ZIP file.



\## Datasets



The data used in this project is published by The Pokémon Company International and compiled by Nimbasa City Post.



The dataset can be found at:



https://www.nimbasacitypost.com/2025/08/pokemon-vgc-2026-season-information.html



An internet connection is required for retrieving online Poképaste data and loading Pokémon sprite images.

