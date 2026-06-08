from pathlib import Path

def delete_run(conn, run_id):
    cursor = conn.cursor()

    cursor.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
    cursor.execute("DELETE FROM timeseries WHERE run_id = ?", (run_id,))
    cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def recompute_metrics(conn, run_id):

    from pde_analysis.processing.extract import extract_timeseries
    from pde_analysis.processing.compute import compute_metrics
    from pde_analysis.db.metrics import add_metric

    cursor = conn.cursor()

    # получаем путь
    cursor.execute("SELECT h5_path, params_json FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()

    h5_path, params_json = row
    if not h5_path:
        raise FileNotFoundError(f"HDF5 path not set for run {run_id}")

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    import json
    params = json.loads(params_json)

    # удаляем старые метрики
    cursor.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))

    # пересчёт
    ts = extract_timeseries(str(h5_path))
    metrics = compute_metrics(ts, params)

    for name, value in metrics.items():
        add_metric(conn, run_id, name, value, version=1)    


def recompute_metrics_for_experiment(conn, experiment_name):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT r.run_id
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE e.name = ?
    """, (experiment_name,))

    run_ids = [row[0] for row in cursor.fetchall()]

    for run_id in run_ids:
        recompute_metrics(conn, run_id)

    conn.commit()