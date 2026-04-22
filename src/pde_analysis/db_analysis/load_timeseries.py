import pandas as pd
from pde_analysis.db.connection import get_connection


def load_timeseries_dataframe(experiment_name: str) -> pd.DataFrame:
    """
    Возвращает DataFrame в long-формате:

    columns:
        run_id, u, Lambda, ts_name, t, value
    """

    conn = get_connection()

    query = """
    SELECT
        r.run_id,
        r.u,
        r.Lambda,
        ts.name AS ts_name,
        ts.t_values,
        ts.y_values
    FROM runs r
    JOIN experiments e ON r.experiment_id = e.experiment_id
    JOIN timeseries ts ON ts.run_id = r.run_id
    WHERE e.name = ?
    """

    df = pd.read_sql_query(query, conn, params=(experiment_name,))
    conn.close()

    if df.empty:
        return df

    # распаковка JSON массивов
    import json

    records = []

    for _, row in df.iterrows():
        t_vals = json.loads(row["t_values"])
        y_vals = json.loads(row["y_values"])

        for t, y in zip(t_vals, y_vals):
            records.append(
                {
                    "run_id": row["run_id"],
                    "u": row["u"],
                    "Lambda": row["Lambda"],
                    "ts_name": row["ts_name"],
                    "t": t,
                    "value": y,
                }
            )

    return pd.DataFrame(records)