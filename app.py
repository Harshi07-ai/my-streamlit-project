"""
Root-level wrapper for Streamlit application.
Allows running 'streamlit run app.py' directly from workspace root.
"""
import os
import sys

# Set working directory to urban_mobility_prediction and run its app.py
project_dir = os.path.join(os.path.dirname(__file__), "urban_mobility_prediction")
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

app_path = os.path.join(project_dir, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

# Execute app.py in global context
exec(compile(code, app_path, 'exec'))
