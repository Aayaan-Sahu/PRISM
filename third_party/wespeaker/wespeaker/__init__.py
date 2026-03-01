"""
Minimal wespeaker package init for model-only embedding use.

WeSep imports `wespeaker.models.speaker_model` for target speaker encoders.
Importing CLI modules here pulls optional diarization deps (`hdbscan`) that are
not required for TSE extraction and may introduce binary ABI issues.
"""

__all__ = []
