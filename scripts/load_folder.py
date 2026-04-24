from pathlib import Path

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


if __name__ == "__main__":
    load_folder(
        folder_path=r"C:\YandexDisk\Yandex.Disk\Ioffe\workspace\one_decay\2D\data\sol\IC3_x0",
        experiment_name="one_decay_parameters_scan_with_IC_dx=0p05",
        recursive=True
    )