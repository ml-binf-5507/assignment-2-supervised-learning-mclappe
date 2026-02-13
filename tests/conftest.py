"""
pytest configuration and fixtures for Assignment 2.

This file sets up the Python path and provides shared fixtures for all tests.
"""

import sys
import os

# Add the project root to sys.path so imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress matplotlib display
import matplotlib
matplotlib.use('Agg')
