"""
CLI entry point for Housing Price Predictor.
"""

import sys
from pathlib import Path

# Add web_app to path for CLI import
sys.path.append(str(Path(__file__).parent.parent / "web_app"))

from cli import main

if __name__ == "__main__":
    main()
