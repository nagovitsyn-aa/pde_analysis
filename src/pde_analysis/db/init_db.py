from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "db" / "simulations.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

    print(f"DB initialized at {DB_PATH}")


if __name__ == "__main__":
    init_db()