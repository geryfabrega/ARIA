"""Entry point for the mutating attack pipeline.

Usage:
    python pipeline/run_attack.py                        # runs on first 5 behaviors
    python pipeline/run_attack.py --behaviors 10         # runs on first N behaviors
    python pipeline/run_attack.py --behaviors 10 --max-cycles 5
    python pipeline/run_attack.py --output outputs/my_results.csv
"""

import argparse
import csv
import os
import sys

from dotenv import load_dotenv

import jailbreakbench as jbb
from attack_pipeline.config import DEFAULT_BEHAVIORS, MAX_CYCLES, OUTPUT_CSV
from attack_pipeline.pipeline import CycleRecord, run_behavior

CSV_COLUMNS = [
    "behavior",
    "cycle",
    "attack_prompt",
    "target_response",
    "jailbroken",
    "judge_reason",
    "feedback",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mutating jailbreak attack pipeline.")
    parser.add_argument(
        "--behaviors",
        type=int,
        default=DEFAULT_BEHAVIORS,
        help=f"Number of JBB behaviors to run (default: {DEFAULT_BEHAVIORS})",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=MAX_CYCLES,
        help=f"Max mutation cycles per behavior (default: {MAX_CYCLES})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    return parser.parse_args()


def record_to_row(r: CycleRecord) -> dict:
    return {
        "behavior": r.behavior,
        "cycle": r.cycle,
        "attack_prompt": r.attack_prompt,
        "target_response": r.target_response,
        "jailbroken": r.jailbroken,
        "judge_reason": r.judge_reason,
        "feedback": r.feedback,
    }


def main() -> None:
    load_dotenv()

    together_api_key = os.environ.get("TOGETHERAI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not together_api_key:
        sys.exit("Error: TOGETHERAI_API_KEY not set in environment or .env file.")
    if not openai_api_key:
        sys.exit("Error: OPENAI_API_KEY not set in environment or .env file.")

    args = parse_args()

    dataset = jbb.read_dataset()
    behaviors = dataset.behaviors[: args.behaviors]
    goals = dataset.goals[: args.behaviors]

    print(f"Running pipeline on {len(behaviors)} behavior(s), max {args.max_cycles} cycle(s) each.")
    print(f"Output → {args.output}\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for i, (behavior, goal) in enumerate(zip(behaviors, goals), start=1):
            print(f"[{i}/{len(behaviors)}] {behavior}")
            try:
                records = run_behavior(
                    goal=goal,
                    behavior=behavior,
                    together_api_key=together_api_key,
                    openai_api_key=openai_api_key,
                    max_cycles=args.max_cycles,
                )
            except Exception as exc:
                print(f"  [pipeline error, skipping behavior] {exc}")
                continue

            for record in records:
                writer.writerow(record_to_row(record))
            csvfile.flush()  # ensure partial results survive crashes

            final = records[-1]
            status = "SUCCESS" if final.jailbroken else "failed"
            print(f"  → {status} after {len(records)} cycle(s)\n")

    print(f"Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
