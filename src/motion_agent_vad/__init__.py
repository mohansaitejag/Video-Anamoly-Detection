"""Motion Agent for Video Anomaly Detection (VAD).

See README.md at the project root for usage. Public entry points live in
scripts/ (inspect_dataset.py, build_baseline.py, run_pipeline.py,
evaluate.py); this package holds the reusable library code.
"""
from .config import Config

__all__ = ["Config"]
__version__ = "0.1.0"
