"""Streamlit Cloud entry point — delegates to dashboard/app.py."""
import runpy
import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_root, "dashboard"))

runpy.run_path(os.path.join(_root, "dashboard", "app.py"), run_name="__main__")
