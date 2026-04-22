import pandas as pd
from pde_analysis.db.connection import get_connection


def load_metrics_dataframe(experiment_name: str) -> pd.DataFrame:
    """
    Загружает метрики эксперимента в wide-формате.

    columns:
        run_id, u, Lambda, <metrics...>
    """

    conn = get_connection()

    query = """
    SELECT
        r.run_id,
        r.u,
        r.Lambda,
        md.name AS metric_name,
        m.value
    FROM runs r
    JOIN experiments e ON r.experiment_id = e.experiment_id
    JOIN metrics m ON m.run_id = r.run_id
    JOIN metric_definitions md ON md.metric_def_id = m.metric_def_id
    WHERE e.name = ?
    """

    df = pd.read_sql_query(query, conn, params=(experiment_name,))

    conn.close()

    if df.empty:
        return df

    df_wide = (
        df.pivot_table(
            index=["run_id", "u", "Lambda"],
            columns="metric_name",
            values="value",
        )
        .reset_index()
    )

    # техническая деталь pandas:
    # после pivot имя колонок хранится в df.columns.name = "metric_name"
    # это иногда мешает (например при сохранении / выводе)
    df_wide.columns.name = None

    return df_wide