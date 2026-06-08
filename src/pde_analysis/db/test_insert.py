import json

from pde_analysis.db.connection import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("INSERT INTO experiments (name) VALUES (?)", ("test_exp",))
experiment_id = cursor.lastrowid

params = {"Lambda": 1.0}

cursor.execute("""
INSERT INTO runs (experiment_id, file_name, h5_path, params_json)
VALUES (?, ?, ?, ?)
""", (
    experiment_id,
    "run_001.h5",
    "data/h5/run_001.h5",
    json.dumps(params)
))

conn.commit()
conn.close()