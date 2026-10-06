#!/usr/bin/env python3
"""Run the full predictability-aware OFDM experiment suite with resumable stages."""
import argparse
from ofdm_mae_cli import PROJECT_DIR, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=PROJECT_DIR / "runs" / "predictability_aware_ofdm", type=str)
    parser.add_argument("--cache", default=PROJECT_DIR / "cache", type=str)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    command = ["all", "--config", PROJECT_DIR / "configs" / "predictability_aware_ofdm.yaml",
               "--output", args.output, "--cache", args.cache]
    if not args.no_resume:
        command.append("--resume")
    run(command)


if __name__ == "__main__":
    main()
