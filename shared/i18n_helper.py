"""Shared i18n helpers for engine scripts (GUI and CLI)."""
import json
from pathlib import Path


def load_locale(locale_dir: Path, lang: str) -> dict:
    path = locale_dir / f"{lang}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def t(loc: dict, key: str, **kwargs) -> str:
    text = loc.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text
