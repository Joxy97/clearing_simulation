import argparse
import json

from .config import load_config
from .runner import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clearing simulation.")
    parser.add_argument("--config", default="configs/default.json", help="Path to config JSON.")
    parser.add_argument("--days", type=int, default=None, help="Override number of days.")
    parser.add_argument("--output", default=None, help="Override output path for metrics JSON.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.days is not None:
        config.setdefault("simulation", {})
        config["simulation"]["n_days"] = args.days
    if args.output is not None:
        config.setdefault("output", {})
        config["output"]["path"] = args.output

    metrics = run_simulation(config)
    summary = metrics.get("summary", {})
    print("Simulation complete.")
    if summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
