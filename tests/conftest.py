"""Make `src/` importable from tests without installing the package.

Experiments use `from src.tascp_eval...` (repo-root on sys.path); tests follow
the same convention so the suite runs from a clean checkout (pytest already
adds rootdir-adjacent paths, but keep it explicit).
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
