def print_summary(data):

    print("=== GRID ===")
    print("t:", data["grid"]["t"].shape)
    print("x:", data["grid"]["x"].shape)
    print("y:", data["grid"]["y"].shape)

    print("\n=== SOLUTION ===")
    for k, v in data["solution"].items():
        print(k, v.shape)

    print("\n=== STEP ===")
    print("dx:", data["step"]["dx"])
    print("dy:", data["step"]["dy"])

    print("\n=== METADATA ===")
    for k, v in data["metadata"].items():
        print(f"{k}: {v}")

    print("\n=== PARAMETERS ===")
    for k, v in data["parameters"].items():
        print(f"{k}: {v}")