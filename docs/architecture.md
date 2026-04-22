# PDE Analysis — Architecture

## 1. Overview

This project processes results of PDE simulations stored in HDF5 files and extracts lightweight analytical data into a SQLite database.

The system is designed to:
- avoid storing heavy data twice
- allow flexible recomputation of derived quantities
- support iterative scientific exploration

---

## 2. Core Principles

- **HDF5 = source of truth**
  - All raw simulation data (fields, grids) are stored only in `.h5` files

- **SQLite = analysis layer**
  - Stores parameters, metadata, time series, and derived metrics

- **Runs are immutable**
  - A run represents a single simulation result
  - Do not modify runs; delete and recreate if needed

- **Metrics are recomputable**
  - Metrics depend on algorithms and may change
  - Old metrics should be deletable and recomputed

- **Time series are lightweight**
  - Only aggregated data is stored (not full fields)

---

## 3. Project Structure

pde_analysis/
│
├── data/
│ ├── h5/ # raw simulation files
│ └── db/ # SQLite database
│
├── notebooks/ # marimo notebooks
│
├── src/pde_analysis/
│ ├── db/ # database access layer
│ ├── processing/ # data extraction + computations
│ ├── pipeline/ # ingestion + management
│
├── scripts/ # one-off scripts (migration, tests)


---

## 4. Database Schema

### experiments

Represents a group of related simulations.

- experiment_id (PK)
- name

---

### runs

Represents a single simulation.

- run_id (PK)
- experiment_id (FK)
- h5_path
- params_json
- status
- note

Denormalized columns (for fast filtering):
- Lambda
- u
- tend
- x0
- rangeX
- rangeY
- dx
- dy

---

### timeseries

Stores derived time-dependent quantities.

- id (PK)
- run_id (FK)
- name
- t_values (JSON)
- y_values (JSON)

Examples:
- E_a(t)
- E_b(t)
- max_a(t)
- max_b(t)

---

### metric_definitions

Defines metric identity and version.

- metric_def_id (PK)
- name
- version
- code_hash (optional)

---

### metrics

Stores scalar results.

- id (PK)
- run_id (FK)
- metric_def_id (FK)
- value (nullable)

---

## 5. Data Flow

HDF5 → extract → timeseries → compute → metrics → SQLite



Steps:

1. Extract parameters and metadata
2. Compute time series from raw fields
3. Compute scalar metrics from time series
4. Store results in database

---

## 6. Extracted Data

### From HDF5

- Parameters (/Parameters)
- Step sizes (/StepSize)
- Metadata (/Metadata)

---

### Time Series

Computed from fields:

- Energy:
  - E_a(t) = ∫ |a|² dx dy
  - E_b(t) = ∫ |b|² dx dy

- Amplitude:
  - max|a|(t)
  - max|b|(t)

All time series are normalized to the first value.

---

## 7. Metrics

Computed from time series (currently for wave `a`):

- growth_rate  
  Exponential increment from best-fit segment in log-space

- amplification  
  Maximum value of normalized signal

- Z  
  log(amplification) / (2π)

- ZtoZ0  
  log(amplification) / (2π * u^-2)

- t_interaction  
  End of exponential growth phase

---

## 8. Algorithms

### Exponential Growth Detection

- Work in log-space: log(y)
- Use prefix sums for fast segment evaluation
- Search over all segments with:
  - minimum length
  - positive slope
- Select segment with highest R²

---

### Interaction Time

Defined as:

- End of the exponential growth segment

Fallback (if no segment found):

- detect drop in local slope of log(y)

---

## 9. Workflow

### Add new run


add_run_from_h5(experiment_name, h5_path)


---

### Recompute metrics

recompute_metrics(run_id)
recompute_metrics_for_experiment(name)


---

### Delete run


delete_run(run_id)


Removes:
- metrics
- timeseries
- run entry

---

## 10. Versioning

Metrics are versioned via:

- metric_definitions.name
- metric_definitions.version

When changing computation logic:

1. Increase version
2. Recompute metrics
3. Keep old values for comparison

---

## 11. Design Decisions

- Store parameters both as JSON and as columns
  - JSON → full record
  - columns → fast filtering

- Store time series as JSON arrays
  - small size
  - easy integration with pandas

- Avoid storing raw fields in database
  - too large
  - already in HDF5

---

## 12. Future Extensions

- Add indices on frequently queried columns
- Add code_hash for metric reproducibility
- Add uncertainty estimation for metrics
- Add automated batch processing
- Improve interaction time detection robustness

---