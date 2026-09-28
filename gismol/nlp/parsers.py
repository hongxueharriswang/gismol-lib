# ============================================================
# FILE: gismol/nlp/parsers.py
# ============================================================
"""Lightweight tokenisation, sentence splitting and rule-based POS tagging."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Token", "Document", "tokenize", "split_sentences", "pos_tag", "parse"]

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9_]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by",
    "and", "or", "but", "if", "then", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those", "as",
    "from", "into", "over", "under", "than", "so", "such",
}

_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "each", "every"}
_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "them", "him", "her"}
_PREPOSITIONS = {"of", "in", "on", "at", "to", "for", "with", "by", "from",
                 "into", "over", "under", "between", "among", "through"}
_CONJUNCTIONS = {"and", "or", "but", "nor", "yet", "so", "because", "although"}
_MODALS = {"must", "shall", "should", "may", "might", "can", "could", "will", "would"}
_COMPARATORS = {"less", "more", "greater", "smaller", "equal", "least", "most"}
_NEGATIONS = {"not", "no", "never", "none", "cannot"}


class Token:
    """A token with rule-based part-of-speech information."""

    __slots__ = ("text", "index", "pos", "lemma", "is_stopword")

    def __init__(self, text: str, index: int = 0, pos: str = "X", lemma: str = ""):
        self.text = text
        self.index = index
        self.pos = pos
        self.lemma = lemma or text.lower()
        self.is_stopword = self.lemma in _STOPWORDS

    def __repr__(self):
        return f"Token({self.text!r}, pos={self.pos!r})"

    def to_dict(self):
        return {"text": self.text, "index": self.index, "pos": self.pos,
                "lemma": self.lemma}


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text or "")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENT.split(text)
    return [p.strip() for p in parts if p.strip()]


def pos_tag(tokens: Sequence[str]) -> List[Token]:
    """Very small rule-based tagger using a closed class lexicon."""
    out: List[Token] = []
    for i, raw in enumerate(tokens):
        low = raw.lower()
        if low in _DETERMINERS:
            pos = "DT"
        elif low in _PRONOUNS:
            pos = "PRP"
        elif low in _PREPOSITIONS:
            pos = "IN"
        elif low in _CONJUNCTIONS:
            pos = "CC"
        elif low in _MODALS:
            pos = "MD"
        elif low in _NEGATIONS:
            pos = "NEG"
        elif low in _COMPARATORS:
            pos = "JJR"
        elif low in ("if", "when", "whenever", "once"):
            pos = "COND"
        elif re.fullmatch(r"-?\d+(\.\d+)?", raw):
            pos = "NUM"
        elif re.fullmatch(r"[<>]=?|=|==|!=|<=|>=", raw):
            pos = "OP"
        elif raw.endswith("ly"):
            pos = "RB"
        elif raw.endswith("ing") or raw.endswith("ed"):
            pos = "VB"
        elif raw[0].isupper():
            pos = "NNP"
        else:
            pos = "NN"
        out.append(Token(raw, i, pos))
    return out


class Document:
    """Parsed text: sentences, tokens and convenience accessors."""

    def __init__(self, text: str, sentences: Optional[Sequence[str]] = None,
                 tokens: Optional[Sequence[Token]] = None):
        self.text = text
        self.sentences = list(sentences or [])
        self.tokens = list(tokens or [])

    @property
    def lemmas(self) -> List[str]:
        return [t.lemma for t in self.tokens]

    @property
    def content_tokens(self) -> List[Token]:
        return [t for t in self.tokens if not t.is_stopword]

    def noun_phrases(self) -> List[str]:
        phrases: List[str] = []
        current: List[str] = []
        for token in self.tokens:
            if token.pos in ("NN", "NNP", "JJ") and not token.is_stopword:
                current.append(token.text)
            else:
                if current:
                    phrases.append(" ".join(current))
                    current = []
        if current:
            phrases.append(" ".join(current))
        return phrases

    def numbers(self) -> List[float]:
        out = []
        for token in self.tokens:
            if token.pos == "NUM":
                try:
                    out.append(float(token.text))
                except ValueError:
                    pass
        return out

    def to_dict(self):
        return {"text": self.text, "sentences": self.sentences,
                "tokens": [t.to_dict() for t in self.tokens]}

    def __repr__(self):
        return f"Document(sentences={len(self.sentences)}, tokens={len(self.tokens)})"


def parse(text: str) -> Document:
    sentences = split_sentences(text)
    tokens = pos_tag(tokenize(text))
    return Document(text, sentences, tokens)
