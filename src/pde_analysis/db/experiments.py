import sqlite3


def get_or_create_experiment(conn: sqlite3.Connection, name: str) -> int:
    cursor = conn.cursor()

    # пробуем найти
    cursor.execute(
        "SELECT experiment_id FROM experiments WHERE name = ?",
        (name,)
    )
    row = cursor.fetchone()

    if row:
        return row[0]

    # создаём
    cursor.execute(
        "INSERT INTO experiments (name) VALUES (?)",
        (name,)
    )

    return cursor.lastrowid