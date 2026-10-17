import os
import csv
from datetime import datetime

# Dynamic path resolution to ensure robustness whether run from root or src
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR) if os.path.basename(_CURRENT_DIR) == "src" else _CURRENT_DIR

# Check data/ directory first, then root fallback
_CANDIDATE_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "landslide_data.csv")
_CANDIDATE_ROOT_PATH = os.path.join(_PROJECT_ROOT, "landslide_data.csv")

if os.path.isfile(_CANDIDATE_DATA_PATH):
    FILE_NAME = _CANDIDATE_DATA_PATH
elif os.path.isfile(_CANDIDATE_ROOT_PATH):
    FILE_NAME = _CANDIDATE_ROOT_PATH
else:
    FILE_NAME = _CANDIDATE_DATA_PATH

BASE_DIR = os.path.dirname(FILE_NAME)

def save_landslide(lat, lon, confidence, location):
    try:
        location = location.replace(",", " ")

        file_exists = os.path.isfile(FILE_NAME)

        with open(FILE_NAME, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(["latitude", "longitude", "confidence", "timestamp", "location"])

            writer.writerow([
                lat,
                lon,
                confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                location
            ])

        print("✅ Saved at:", FILE_NAME)  # DEBUG LINE

    except Exception as e:
        print("❌ Error saving:", e)