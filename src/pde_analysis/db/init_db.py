from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "db" / "simulations.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def has_column(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())


def migrate_runs_table(cursor):
    cursor.execute("ALTER TABLE runs ADD COLUMN file_name TEXT")

    cursor.execute("SELECT run_id, h5_path FROM runs")
    rows = cursor.fetchall()
    for run_id, h5_path in rows:
        if h5_path:
            cursor.execute(
                "UPDATE runs SET file_name = ? WHERE run_id = ?",
                (Path(h5_path).name, run_id)
            )

    cursor.execute("DROP INDEX IF EXISTS idx_runs_h5_path")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_file_name ON runs(file_name)")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if has_column(cursor, "runs", "file_name"):
        with open(SCHEMA_PATH, "r") as f:
            cursor.executescript(f.read())
    else:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
        if cursor.fetchone() is None:
            with open(SCHEMA_PATH, "r") as f:
                cursor.executescript(f.read())
        else:
            migrate_runs_table(cursor)
            with open(SCHEMA_PATH, "r") as f:
                cursor.executescript(f.read())

    conn.commit()
    conn.close()

    print(f"DB initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()