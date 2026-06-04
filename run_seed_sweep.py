import subprocess
import sys


def main() -> None:
    methods = ["PG", "DPG", "WPO"]
    seeds = range(41, 46)

    total = len(methods) * len(list(seeds))
    idx = 0

    for seed in range(41, 46):
        for method in methods:
            idx += 1
            cmd = [
                sys.executable,
                "train.py",
                "--method",
                method,
                "--num-iters",
                "30000",
                "--gaussian-fisher-scaling",
                "wpo",
                "--seed",
                str(seed),
            ]
            print(f"[{idx}/{total}] Running: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)

    print("All runs completed.")


if __name__ == "__main__":
    main()
