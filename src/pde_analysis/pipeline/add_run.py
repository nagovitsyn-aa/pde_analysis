from pathlib import Path

from pde_analysis.db.connection import get_connection
from pde_analysis.db.experiments import get_or_create_experiment
from pde_analysis.db.runs import create_run
from pde_analysis.db.timeseries import add_timeseries

from pde_analysis.processing.extract import extract_params, extract_metadata, extract_timeseries
from pde_analysis.processing.compute import compute_derived_timeseries

import json


def run_exists(conn, file_name):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT run_id FROM runs WHERE file_name = ?",
        (file_name,)
    )
    return cursor.fetchone()


def add_run_from_h5(
    experiment_name: str,
    h5_path: str | Path,
    dimension: str,
    description: str | None = None,
    decay_type: str = "single",
    file_prefix: str | None = None,
):
    """
    Главная функция добавления одного расчёта в БД.

    Параметры
    ---------
    experiment_name : str
        Имя серии расчётов.
    h5_path : str | Path
        Путь к HDF5-файлу.
    dimension : str
        Размерность ('2D', '1Dx', '1Dz').
    description : str | None
        Описание серии (сохраняется в experiments.description).
    decay_type : str
        Тип распада: 'single', 'double', 'absolute'.
    file_prefix : str | None
        Префикс для file_name в БД (чтобы избежать коллизий同名ных .h5
        из разных папок). Например 'default_IC'. Итоговое имя:
        "{prefix}/{original_name}".

    Сохраняет:
      - параметры расчёта
      - метаданные
      - сырые временные ряды (из extract_timeseries)
      - нормированные временные ряды (из compute_derived_timeseries)

    Метрики НЕ вычисляются при загрузке — для этого есть compute_metrics(),
    вызываемый отдельно (из ноутбуков, manage.py и т.д.).
    """
    conn = get_connection()
    h5_path = Path(h5_path)

    # --- 0. базовая проверка ---
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    file_name = h5_path.name
    if file_prefix:
        file_name = f"{file_prefix}/{file_name}"
    existing = run_exists(conn, file_name)

    if existing:
        print(f"[SKIP] run already exists: {file_name}")
        return None

    # --- 1. извлечение данных ---
    params = extract_params(h5_path)
    metadata = extract_metadata(h5_path)
    status = metadata.get("CalculationStatus", "unknown")
    note = json.dumps(metadata, ensure_ascii=False, default=str)

    raw_ts = extract_timeseries(h5_path)

    # --- 2. нормированные ряды ---
    derived_ts = compute_derived_timeseries(raw_ts)

    # --- 3. запись в БД ---
    try:
        # 3.1 эксперимент
        experiment_id = get_or_create_experiment(
            conn, experiment_name,
            description=description,
            decay_type=decay_type,
        )

        # 3.2 run
        run_id = create_run(
            conn=conn,
            experiment_id=experiment_id,
            params=params,
            file_name=file_name,
            dimension=dimension,
            h5_path=str(h5_path),
            status=status,
            note=note,
        )

        # 3.3 сырые временные ряды
        for name, (t, y) in raw_ts.items():
            add_timeseries(conn, run_id, name, t, y)

        # 3.4 нормированные временные ряды
        for name, (t, y) in derived_ts.items():
            add_timeseries(conn, run_id, name, t, y)

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        conn.close()

    return run_id
