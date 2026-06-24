CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    experiment_id INTEGER NOT NULL,

    file_name TEXT NOT NULL,
    h5_path TEXT,
    dimension TEXT NOT NULL,
    params_json TEXT NOT NULL,

    Lambda REAL,
    u REAL,
    tend REAL,
    x0 REAL,
    rangeX REAL,
    rangeY REAL,
    dx REAL,
    dy REAL,

    status TEXT DEFAULT 'success',
    note TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE UNIQUE INDEX idx_runs_file_name ON runs(file_name);

CREATE TABLE IF NOT EXISTS metric_definitions (
    metric_def_id INTEGER PRIMARY KEY,

    name TEXT NOT NULL,
    version INTEGER NOT NULL,

    code_hash TEXT,

    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY,

    run_id INTEGER NOT NULL,
    metric_def_id INTEGER NOT NULL,

    value REAL ,

    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (metric_def_id) REFERENCES metric_definitions(metric_def_id)
);

CREATE TABLE IF NOT EXISTS timeseries (
    series_id INTEGER PRIMARY KEY,

    run_id INTEGER NOT NULL,
    name TEXT NOT NULL,

    t_values TEXT NOT NULL,
    y_values TEXT NOT NULL,

    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment
ON runs(experiment_id);