# ============================================================
# FILE: gismol/nlp/__init__.py
# ============================================================
"""Natural-language interface to COH: parsing, constraint extraction,
text-to-COH compilation, dialogue management and validation."""

from .normalizers import TextNormalizer, normalize
from .parsers import Document, Token, parse, split_sentences, tokenize
from .entity_extractor import Entity, EntityExtractor, extract_entities
from .relations_miner import Relation, RelationMiner, mine_relations
from .intent_recognizer import Intent, IntentRecognizer
from .constraint_parser import ConstraintParser, ParsedConstraints, parse_constraints
from .embedders import (
    BagOfWordsEmbedder,
    CharNGramEmbedder,
    HashingEmbedder,
    TfidfEmbedder,
)
from .similarity import (
    SimilarityIndex,
    cosine_similarity,
    jaccard,
    levenshtein,
)
from .response_generator import ResponseGenerator
from .chat_module import ChatModule, ChatResponse
from .chat_manager import ChatManager, ChatSession
from .text_to_coh import TextToCOH, text_to_coh
from .validation import COHValidator, ValidationIssue, validate_coh

__all__ = [
    "BagOfWordsEmbedder",
    "COHValidator",
    "CharNGramEmbedder",
    "ChatManager",
    "ChatModule",
    "ChatResponse",
    "ChatSession",
    "ConstraintParser",
    "Document",
    "Entity",
    "EntityExtractor",
    "HashingEmbedder",
    "Intent",
    "IntentRecognizer",
    "ParsedConstraints",
    "Relation",
    "RelationMiner",
    "ResponseGenerator",
    "SimilarityIndex",
    "TextNormalizer",
    "TextToCOH",
    "TfidfEmbedder",
    "Token",
    "ValidationIssue",
    "cosine_similarity",
    "extract_entities",
    "jaccard",
    "levenshtein",
    "mine_relations",
    "normalize",
    "parse",
    "parse_constraints",
    "split_sentences",
    "text_to_coh",
    "tokenize",
    "validate_coh",
]