"""
evals/db.py - Throwaway databases for the DB-backed tests and the SQL eval.

Given a superuser URL to a maintenance database (e.g. .../postgres), create
a fresh database, apply the same SQL files docker compose applies
(schema.sql, schema_vectors.sql, agent_views.sql, agent_role.sql), set the
agent_reader password, load the synthetic seed, and hand back two URLs:
the admin one and the least-privilege agent one. Drop it afterwards.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import psycopg2
from psycopg2 import sql as pgsql
from psycopg2.extensions import make_dsn

CORE = Path(__file__).resolve().parents[1]
DB_DIR = CORE / "eo_brain" / "db"
SCHEMA_FILES = [
    DB_DIR / "schema.sql",
    DB_DIR / "schema_vectors.sql",
    DB_DIR / "agent_views.sql",
    DB_DIR / "agent_role.sql",
]
SEED_FILE = CORE / "evals" / "fixtures" / "synthetic_seed.sql"
AGENT_ROLE = "agent_reader"


@dataclass(frozen=True)
class ScratchDB:
    name: str
    admin_dsn: str   # superuser, this database
    agent_dsn: str   # agent_reader, this database


def _admin(dsn: str):
    conn = psycopg2.connect(dsn, connect_timeout=5)
    conn.autocommit = True
    return conn


def create_scratch_db(
    admin_url: str,
    *,
    seed: bool = True,
    prefix: str = "agent_scratch",
    files: Optional[Sequence[Path]] = None,
) -> ScratchDB:
    """
    ``files`` replaces SCHEMA_FILES, applied in order (the upgrade tests build a
    database from older copies of them); the agent_reader password is set after
    them, so the last ones must create the role.
    """
    name = f"{prefix}_{secrets.token_hex(4)}"
    password = secrets.token_urlsafe(16)
    conn = _admin(admin_url)
    try:
        with conn.cursor() as cur:
            cur.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(name)))
    finally:
        conn.close()

    admin_dsn = make_dsn(admin_url, dbname=name)
    agent_dsn = make_dsn(admin_url, dbname=name, user=AGENT_ROLE, password=password)
    db = ScratchDB(name=name, admin_dsn=admin_dsn, agent_dsn=agent_dsn)
    try:
        conn = _admin(admin_dsn)
        try:
            with conn.cursor() as cur:
                for path in (SCHEMA_FILES if files is None else files):
                    cur.execute(path.read_text(encoding="utf-8"))
                cur.execute(
                    pgsql.SQL("ALTER ROLE {} PASSWORD %s").format(pgsql.Identifier(AGENT_ROLE)), (password,)
                )
                if seed:
                    cur.execute(SEED_FILE.read_text(encoding="utf-8"))
        finally:
            conn.close()
    except BaseException:
        drop_scratch_db(admin_url, db)  # do not leak half-built databases
        raise
    return db


def drop_scratch_db(admin_url: str, db: ScratchDB) -> None:
    conn = _admin(admin_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                (db.name,),
            )
            cur.execute(pgsql.SQL("DROP DATABASE IF EXISTS {}").format(pgsql.Identifier(db.name)))
    finally:
        conn.close()


def fingerprint(admin_dsn: str) -> Dict[str, Optional[str]]:
    """Content hash of the application tables and the list of relations: must not change."""
    conn = _admin(admin_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY id), '')) FROM jobs t")
            jobs = cur.fetchone()
            cur.execute(
                "SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY id), '')) FROM alert_features t"
            )
            feats = cur.fetchone()
            cur.execute(
                "SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY job_id, bin_index), '')) "
                "FROM index_value_bins t"
            )
            bins = cur.fetchone()
            cur.execute(
                "SELECT count(*), md5(coalesce(string_agg(t::text, '|' ORDER BY job_id, severity, quadrant), '')) "
                "FROM severity_class_areas t"
            )
            classes = cur.fetchone()
            cur.execute(
                "SELECT md5(string_agg(n.nspname || '.' || c.relname || ':' || c.relkind::text, ',' "
                "ORDER BY n.nspname, c.relname)) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')"
            )
            rels = cur.fetchone()
            cur.execute(
                "SELECT md5(coalesce(string_agg(r.rolname || ':' || coalesce(array_to_string(r.rolconfig, ','), ''), "
                "';' ORDER BY r.rolname), '')) FROM pg_roles r WHERE r.rolname = %s",
                (AGENT_ROLE,),
            )
            role = cur.fetchone()
    finally:
        conn.close()
    return {
        "jobs": f"{jobs[0]}:{jobs[1]}",
        "alert_features": f"{feats[0]}:{feats[1]}",
        "index_value_bins": f"{bins[0]}:{bins[1]}",
        "severity_class_areas": f"{classes[0]}:{classes[1]}",
        "relations": rels[0],
        "agent_role": role[0],
    }
