from argparse import ArgumentParser
from pathlib import Path

import yaml

from pde_analysis.pipeline.add_run import add_run_from_h5


def load_folder(folder_path, experiment_name, dimension, recursive=True):
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    files = sorted(folder.rglob("*.h5") if recursive else folder.glob("*.h5"))

    print(f"Found {len(files)} files")

    success = 0
    skipped = 0
    failed = 0

    for f in files:
        try:
            run_id = add_run_from_h5(
                experiment_name=experiment_name,
                h5_path=f,
                dimension=dimension,
            )

            if run_id is None:
                skipped += 1
                print(f"[SKIP] {f.name}")
            else:
                success += 1
                print(f"[OK] {f.name}")

        except Exception as e:
            failed += 1
            print(f"[FAIL] {f}: {e}")

    print("\n--- SUMMARY ---")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")


def load_config(config_path):
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_file} must contain a YAML mapping")

    return config


def main():
    parser = ArgumentParser(description="Load HDF5 runs into the simulations database")

    parser.add_argument("--config", help="YAML ingestion config file")

    parser.add_argument("--folder", help="Folder containing .h5 files")
    parser.add_argument("--experiment", help="Experiment name for imported runs")
    parser.add_argument("--dimension", choices=["1D", "2D"], help="Run dimension")

    parser.add_argument("--recursive", action="store_true", help="Force recursive search")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive search")

    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}

    if args.recursive and args.no_recursive:
        parser.error("Cannot use --recursive and --no-recursive together")

    folder_path = args.folder or config.get("folder")
    experiment_name = args.experiment or config.get("experiment")
    dimension = args.dimension or config.get("dimension")

    if args.recursive:
        recursive = True
    elif args.no_recursive:
        recursive = False
    else:
        recursive = config.get("recursive", True)

    if folder_path is None:
        parser.error("folder must be provided via CLI or config")
    if experiment_name is None:
        parser.error("experiment must be provided via CLI or config")
    if dimension is None:
        parser.error("dimension must be provided via CLI or config")

    load_folder(
        folder_path=folder_path,
        experiment_name=experiment_name,
        dimension=dimension,
        recursive=recursive,
    )


if __name__ == "__main__":
    main()