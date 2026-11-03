"""E1 runner scaffold."""

import argparse

EXPERIMENT_ID = "E1"
STATUS = "NOT_IMPLEMENTED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"{EXPERIMENT_ID} DRY_RUN config={args.config} output={args.output_dir} seed={args.seed}")
        return 0
    parser.error(f"{EXPERIMENT_ID} is {STATUS}; use --dry-run until implemented")


if __name__ == "__main__":
    raise SystemExit(main())

