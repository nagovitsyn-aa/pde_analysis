from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "db" / "simulations.db"

def get_connection():
    return sqlite3.connect(DB_PATH)