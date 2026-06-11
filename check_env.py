import os

print("=== Checking .env file ===")

env_path = ".env"
if not os.path.exists(env_path):
    print("ERROR: .env file not found!")
else:
    with open(env_path) as f:
        lines = f.readlines()
    print(f"Total lines in .env: {len(lines)}")
    print()
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            # Remove quotes if present
            v_clean = v.strip('"').strip("'")
            print(f"Line {i}: KEY={k}")
            print(f"         VALUE=[{v[:30]}...]")
            print(f"         Length={len(v_clean)}")
            if v != v_clean:
                print(f"         ⚠️  WARNING: Value has quotes — remove them!")
            print()
        else:
            print(f"Line {i}: SKIPPED (no = sign): {line}")