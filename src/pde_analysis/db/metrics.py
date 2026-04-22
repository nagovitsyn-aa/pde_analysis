import sqlite3


def get_or_create_metric_def(
    conn: sqlite3.Connection,
    name: str,
    version: int,
    code_hash: str | None = None
) -> int:

    cursor = conn.cursor()

    cursor.execute("""
        SELECT metric_def_id
        FROM metric_definitions
        WHERE name = ? AND version = ?
    """, (name, version))

    row = cursor.fetchone()

    if row:
        return row[0]

    cursor.execute("""
        INSERT INTO metric_definitions (name, version, code_hash)
        VALUES (?, ?, ?)
    """, (name, version, code_hash))

    return cursor.lastrowid


def add_metric(
    conn: sqlite3.Connection,
    run_id: int,
    name: str,
    value: float,
    version: int = 1
):
    cursor = conn.cursor()

    metric_def_id = get_or_create_metric_def(
        conn,
        name=name,
        version=version
    )
    if value is None:
        print(f"[WARN] metric {name} is None for run {run_id}")
        
    cursor.execute("""
        INSERT INTO metrics (run_id, metric_def_id, value)
        VALUES (?, ?, ?)
    """, (
        run_id,
        metric_def_id,
        None if value is None else float(value)
))