"""
GeoSentinel — Backward Compatibility Runner
Ensures that existing scripts, deployment pipelines, or users running `streamlit run app1.py`
continue to execute the main application seamlessly.
"""
import os
import runpy

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    runpy.run_path(app_path, run_name="__main__")
