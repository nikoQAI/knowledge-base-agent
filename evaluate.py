#!/usr/bin/env python3
"""Entry point — run from project root."""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "CLI"))
runpy.run_module("evaluate", run_name="__main__")
