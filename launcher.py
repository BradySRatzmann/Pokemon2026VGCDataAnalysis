
import sys
import time
import webbrowser
import shutil

from pathlib import Path
from streamlit.web import cli as stcli

def export_data_file():
    # Get the folder containing the executable
    if getattr(sys, "frozen", False):
        exe_folder = Path(sys.executable).parent
        bundled_data = Path(sys._MEIPASS) / "data.txt"
    else:
        exe_folder = Path(__file__).parent
        bundled_data = Path(__file__).parent / "data.txt"

    destination = exe_folder / "data.txt"

    # Copy data.txt if it doesn't already exist
    if bundled_data.exists() and not destination.exists():
        shutil.copy2(bundled_data, destination)

def get_app_path():
    if getattr(sys, "frozen", False):
        # Running as a packaged executable
        return Path(sys._MEIPASS) / "app.py"

    # Running normally in Python
    return Path(__file__).parent / "app.py"


if __name__ == "__main__":
    export_data_file()

    app_path = get_app_path()

    # Open the browser shortly after the server starts
    import threading

    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")

    threading.Thread(
        target=open_browser,
        daemon=True
    ).start()

    # Run Streamlit directly in the main thread
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--server.port=8501",
        "--global.developmentMode=false"
    ]

    stcli.main()
