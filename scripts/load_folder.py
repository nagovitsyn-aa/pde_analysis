from argparse import ArgumentParser
from pathlib import Path

import yaml

from pde_analysis.pipeline.add_run import add_run_from_h5


def load_folder(folder_path, experiment_name, recursive=True):
    folder = Path(folder_path)

    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    if recursive:
        files = folder.rglob("*.h5")
    else:
        files = folder.glob("*.h5")

    files = sorted(list(files))

    print(f"Found {len(files)} files")

    success = 0
    skipped = 0
    failed = 0

    for f in files:
        try:
            run_id = add_run_from_h5(
                experiment_name=experiment_name,
                h5_path=f
            )

            if run_id is None:
                skipped += 1
            else:
                success += 1

            print(f"[OK] {f}")

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


if __name__ == "__main__":
    parser = ArgumentParser(description="Load HDF5 runs into the simulations database")
    parser.add_argument("--config", help="YAML ingestion config file")
    parser.add_argument("folder_path", nargs="?", help="Folder containing .h5 files")
    parser.add_argument("experiment_name", nargs="?", help="Experiment name for imported runs")
    parser.add_argument("--recursive", action="store_true", help="Force recursive search")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive search")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}

    if args.recursive and args.no_recursive:
        parser.error("Cannot use --recursive and --no-recursive together")

    folder_path = args.folder_path or config.get("folder")
    experiment_name = args.experiment_name or config.get("experiment")
    if args.recursive:
        recursive = True
    elif args.no_recursive:
        recursive = False
    else:
        recursive = config.get("recursive", True)

    if folder_path is None or experiment_name is None:
        parser.error("folder_path and experiment_name must be provided via CLI or config")

    load_folder(
        folder_path=folder_path,
        experiment_name=experiment_name,
        recursive=recursive,
    )