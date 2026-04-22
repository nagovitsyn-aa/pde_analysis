from pathlib import Path

from pde_analysis.db.connection import get_connection
from pde_analysis.db.experiments import get_or_create_experiment
from pde_analysis.db.runs import create_run
from pde_analysis.db.metrics import add_metric
from pde_analysis.db.timeseries import add_timeseries

from pde_analysis.processing.extract import extract_params, extract_metadata, extract_timeseries
from pde_analysis.processing.compute import compute_metrics

import json


def run_exists(conn, h5_path):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT run_id FROM runs WHERE h5_path = ?",
        (h5_path,)
    )
    return cursor.fetchone()



def add_run_from_h5(experiment_name: str, h5_path: str | Path):
    """
    Главная функция добавления одного расчёта в БД.
    """
    conn = get_connection()
    h5_path = Path(h5_path)

    
    # --- 0. базовая проверка ---
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    
    existing = run_exists(conn, str(h5_path))

    if existing:
        print(f"[SKIP] run already exists: {h5_path}")
        return existing[0]

    # --- 1. извлечение данных ---
    params = extract_params(h5_path)
    metadata = extract_metadata(h5_path)
    status = metadata.get("CalculationStatus", "unknown")
    note = json.dumps(metadata)

    timeseries_dict = extract_timeseries(h5_path)

    # --- 2. вычисление метрик ---
    metrics_dict = compute_metrics(timeseries_dict, params)

    # --- 3. запись в БД ---
    
    try:
        # 3.1 эксперимент
        experiment_id = get_or_create_experiment(conn, experiment_name)

        # 3.2 run
        run_id = create_run(
        conn=conn,
        experiment_id=experiment_id,
        params=params,
        h5_path=str(h5_path),
        status=status,
        note=note
)

        # 3.3 временные ряды
        for name, (t, y) in timeseries_dict.items():
            add_timeseries(conn, run_id, name, t, y)

        # 3.4 метрики
        for name, value in metrics_dict.items():
            add_metric(conn, run_id, name, value, version=1)

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        conn.close()

    return run_id