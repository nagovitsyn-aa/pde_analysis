import sqlite3
import json


def add_timeseries(
    conn: sqlite3.Connection,
    run_id: int,
    name: str,
    t,
    y
):
    cursor = conn.cursor()

    # предполагаем numpy arrays
    t_list = list(map(float, t))
    y_list = list(map(float, y))

    cursor.execute("""
        INSERT INTO timeseries (run_id, name, t_values, y_values)
        VALUES (?, ?, ?, ?)
    """, (
        run_id,
        name,
        json.dumps(t_list),
        json.dumps(y_list)
    ))