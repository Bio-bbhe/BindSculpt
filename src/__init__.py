# src/__init__.py
import os
# Set PROJECT_ROOT environment variable (useful for config paths)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.environ['PROJECT_ROOT'] = PROJECT_ROOT

__version__ = "0.1.0"
