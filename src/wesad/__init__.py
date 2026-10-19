"""WESAD stress-detection pipeline: loading, validation, feature extraction, analysis and plotting.

Modules
-------
config    PipelineConfig (every setting) and fixed WESAD facts
loading   load/validate raw participant files and window-level feature tables
features  raw signals -> one row per 60-second window
analysis  baseline normalization and effect sizes
plots     figure helpers
"""
from .config import PipelineConfig  # noqa: F401
