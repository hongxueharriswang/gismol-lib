# ============================================================
# FILE: gismol/nlp/normalizers.py
# ============================================================
"""Text normalisation utilities."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional, Sequence

__all__ = ["normalize", "TextNormalizer", "remove_accents", "strip_punctuation"]

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\.\,\-\+\*\/\=\<\>\!\?\:\;]")
_MULTI_PUNCT = re.compile(r"([.!?]){2,}")


def remove_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def strip_punctuation(text: str, keep: str = ".,-+*/=<>!?:;") -> str:
    return _PUNCT.sub(" ", text)


def normalize(text: str, lowercase: bool = True, unicode_form: str = "NFKC",
              collapse_whitespace: bool = True, strip_accents: bool = False,
              remove_symbols: bool = False) -> str:
    """Canonical text normalisation."""
    if text is None:
        return ""
    out = str(text)
    if unicode_form:
        out = unicodedata.normalize(unicode_form, out)
    if strip_accents:
        out = remove_accents(out)
    if lowercase:
        out = out.lower()
    if remove_symbols:
        out = strip_punctuation(out)
    if collapse_whitespace:
        out = _WS.sub(" ", out).strip()
    out = _MULTI_PUNCT.sub(r"\1", out)
    return out


class TextNormalizer:
    """Configurable normaliser."""

    def __init__(self, lowercase: bool = True, strip_accents: bool = False,
                 remove_symbols: bool = False, unicode_form: str = "NFKC"):
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.remove_symbols = remove_symbols
        self.unicode_form = unicode_form

    def __call__(self, text: str) -> str:
        return normalize(text, self.lowercase, self.unicode_form,
                         strip_accents=self.strip_accents,
                         remove_symbols=self.remove_symbols)

    def __repr__(self):
        return (f"TextNormalizer(lowercase={self.lowercase}, "
                f"strip_accents={self.strip_accents}, "
                f"remove_symbols={self.remove_symbols})")
    