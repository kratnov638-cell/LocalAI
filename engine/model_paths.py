"""Model-directory resolution shared by the UI and model discovery code."""
from __future__ import annotations

import os
from pathlib import Path


def default_models_dir(project_root: str | Path) -> Path:
    """Return the preferred GGUF directory without moving or copying weights.

    LOCALAI_MODELS_DIR overrides the default. Otherwise prefer the user's
    external Windows model folder when present, and fall back to project/models
    for existing installations that have not moved their files.
    """
    override = os.environ.get('LOCALAI_MODELS_DIR', '').strip().strip('"')
    if override:
        return Path(override).expanduser().resolve()
    external = Path(r'C:\LLM')
    if external.is_dir():
        return external
    return (Path(project_root).resolve() / 'models')


def discover_gguf_models(project_root: str | Path) -> list[Path]:
    """List GGUF files from the configured directory, sorted for stable UI use."""
    directory = default_models_dir(project_root)
    if not directory.is_dir():
        return []
    return sorted((p.resolve() for p in directory.glob('*.gguf') if p.is_file()), key=lambda p: p.name.casefold())
