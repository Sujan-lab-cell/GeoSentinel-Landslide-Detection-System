import os
import csv
from datetime import datetime

# 🔥 FIXED PATH (very important)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_NAME = os.path.join(BASE_DIR, "landslide_data.csv")

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