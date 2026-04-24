"""Streamlit Cloud entry point — delegates to dashboard/app.py."""
import runpy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dashboard"))
os.chdir(os.path.join(os.path.dirname(__file__), "dashboard"))

runpy.run_path("dashboard/app.py", run_name="__main__")
