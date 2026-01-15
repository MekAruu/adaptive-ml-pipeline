from pathlib import Path
import argparse

from baseline import run_all, run_static, run_periodic, run_drift

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "static", "periodic", "drift"], default="all")
    args = parser.parse_args()

    results_dir = Path("results")
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.mode in ["all", "static"]:
        run_static(results_dir)

    if args.mode in ["all", "periodic"]:
        run_periodic(results_dir, retrain_every=5)

    if args.mode in ["all", "drift"]:
        run_drift(results_dir, window=200, step=50, drift_threshold=0.25)

    print("Done. Check /results for metrics and figures.")

if __name__ == "__main__":
    main()
