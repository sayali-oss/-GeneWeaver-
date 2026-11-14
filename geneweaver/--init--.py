"""GeneWeaver: GPU-accelerated CRISPR guide RNA off-target alignment engine."""
from .aligner import search_guide, search_guides
from .backend import get_backend

__all__ = ["search_guide", "search_guides", "get_backend"]
__version__ = "1.0.0"
