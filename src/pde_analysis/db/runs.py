import json


def create_run(
    conn,
    experiment_id,
    params,
    file_name,
    h5_path=None,
    status="success",
    note=None
):
    cursor = conn.cursor()

    p = params["Parameters"]
    step = params["Step"]

    cursor.execute("""
        INSERT INTO runs (
            experiment_id,
            file_name,
            h5_path,
            params_json,
            status,
            note,
            Lambda,
            u,
            tend,
            x0,
            rangeX,
            rangeY,
            dx,
            dy
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        experiment_id,
        file_name,
        h5_path,
        json.dumps(params),
        status,
        note,

        float(p.get("Lambda", None)),
        float(p.get("u", None)),
        float(p.get("tend", None)),
        float(p.get("x0", None)),

        float(p.get("rangeX", None)),
        float(p.get("rangeY", None)),

        float(step.get("dx", None)),
        float(step.get("dy", None))
    ))

    return cursor.lastrowid