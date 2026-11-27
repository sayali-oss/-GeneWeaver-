"""Row models for the SQLite store."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalysisRow:
    id: str
    name: str
    timestamp: str
    system: str
    engine_used: str
    n_references: int
    n_guides: int
    total_bp: int
    perfect_targets: int
    off_targets: int
    avg_alignment_score: float
    total_s: float
    reference_ids: str
    guide_ids: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    system TEXT, engine_used TEXT,
    n_references INTEGER, n_guides INTEGER, total_bp INTEGER,
    perfect_targets INTEGER, off_targets INTEGER,
    avg_alignment_score REAL, total_s REAL,
    reference_ids TEXT, guide_ids TEXT,
    results_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    gpu_used INTEGER,
    results_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_ts ON analyses(timestamp);
"""
