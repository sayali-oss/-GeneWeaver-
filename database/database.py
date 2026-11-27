"""SQLite persistence (standard-library sqlite3).  Completed analyses are never overwritten."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

import config
from database.models import SCHEMA


@contextmanager
def connect(path=None):
    conn = sqlite3.connect(str(path or config.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_analysis(results: dict, path=None) -> None:
    s = results["summary"]
    with connect(path) as c:
        c.execute(
            "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (results["id"], results["name"], results["timestamp"], results["config"]["system"],
             results["engine"]["used"], s["n_references"], s["n_guides"], s["total_bp"],
             s["perfect_targets"], s["off_targets"], s["avg_alignment_score"],
             results["timings"]["total_s"],
             ", ".join(r["id"] for r in results["references"]),
             ", ".join(g["id"] for g in results["guides"]),
             json.dumps(results)))


def list_analyses(search: str = "", engine: str = "", system: str = "", path=None) -> list:
    q = ("SELECT id,name,timestamp,system,engine_used,n_references,n_guides,total_bp,perfect_targets,"
         "off_targets,avg_alignment_score,total_s,reference_ids,guide_ids FROM analyses WHERE 1=1")
    args = []
    if search:
        q += " AND (name LIKE ? OR reference_ids LIKE ? OR guide_ids LIKE ?)"
        args += [f"%{search}%"] * 3
    if engine:
        q += " AND engine_used = ?"; args.append(engine)
    if system:
        q += " AND system = ?"; args.append(system)
    q += " ORDER BY timestamp DESC"
    with connect(path) as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def get_analysis(analysis_id: str, path=None):
    with connect(path) as c:
        row = c.execute("SELECT results_json FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    return json.loads(row["results_json"]) if row else None


def latest_analysis(path=None):
    with connect(path) as c:
        row = c.execute("SELECT results_json FROM analyses ORDER BY timestamp DESC, rowid DESC LIMIT 1").fetchone()
    return json.loads(row["results_json"]) if row else None


def delete_analysis(analysis_id: str, path=None) -> None:
    with connect(path) as c:
        c.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))


def aggregate_stats(path=None) -> dict:
    with connect(path) as c:
        r = c.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(n_references),0) refs, COALESCE(SUM(n_guides),0) guides,"
            " COALESCE(SUM(perfect_targets),0) targets, COALESCE(SUM(off_targets),0) offs,"
            " COALESCE(AVG(avg_alignment_score),0) avg_score FROM analyses").fetchone()
    return dict(r)


def save_benchmark(bench: dict, path=None) -> int:
    with connect(path) as c:
        cur = c.execute("INSERT INTO benchmarks (timestamp, gpu_used, results_json) VALUES (?,?,?)",
                        (bench["timestamp"], int(bool(bench.get("gpu_used"))), json.dumps(bench)))
        return cur.lastrowid


def latest_benchmark(path=None):
    with connect(path) as c:
        row = c.execute("SELECT results_json FROM benchmarks ORDER BY id DESC LIMIT 1").fetchone()
    return json.loads(row["results_json"]) if row else None
