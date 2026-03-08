"""Entry point for the mutating attack pipeline.

Usage:
    python pipeline_colab/run_attack.py                  # runs on first 10 behaviors
    python pipeline_colab/run_attack.py --behaviors 10   # runs on first N behaviors
    python pipeline_colab/run_attack.py --behaviors 10 --max-cycles 5
    python pipeline_colab/run_attack.py --output outputs/my_results.csv
"""

import argparse
import csv
import os
import sys

from dotenv import load_dotenv

import jailbreakbench as jbb
from attack_pipeline.config import (
    DEFAULT_BEHAVIORS,
    FINAL_PROMPTS_CSV,
    MAX_CYCLES,
    OUTPUT_CSV,
    validate_attack_workflow_models,
)
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

FINAL_PROMPT_COLUMNS = [
    "behavior",
    "goal",
    "final_cycle",
    "final_attack_prompt",
    "final_jailbroken",
    "final_judge_reason",
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
    parser.add_argument(
        "--final-prompts-output",
        type=str,
        default=FINAL_PROMPTS_CSV,
        help=f"Final prompts CSV path (default: {FINAL_PROMPTS_CSV})",
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

    validate_attack_workflow_models()
    model_api_key = os.environ.get("MODEL_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not openai_api_key:
        sys.exit("Error: OPENAI_API_KEY not set in environment or .env file.")

    args = parse_args()

    dataset = jbb.read_dataset()
    behaviors = dataset.behaviors[: args.behaviors]
    goals = dataset.goals[: args.behaviors]

    print(f"Running pipeline on {len(behaviors)} behavior(s), max {args.max_cycles} cycle(s) each.")
    print(f"Output → {args.output}\n")
    print(f"Final prompts output → {args.final_prompts_output}\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.final_prompts_output) or ".", exist_ok=True)

    with (
        open(args.output, "w", newline="", encoding="utf-8") as csvfile,
        open(args.final_prompts_output, "w", newline="", encoding="utf-8") as final_csvfile,
    ):
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        final_writer = csv.DictWriter(final_csvfile, fieldnames=FINAL_PROMPT_COLUMNS)
        writer.writeheader()
        final_writer.writeheader()

        for i, (behavior, goal) in enumerate(zip(behaviors, goals), start=1):
            print(f"[{i}/{len(behaviors)}] {behavior}")
            try:
                records = run_behavior(
                    goal=goal,
                    behavior=behavior,
                    model_api_key=model_api_key,
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
            final_writer.writerow(
                {
                    "behavior": behavior,
                    "goal": goal,
                    "final_cycle": final.cycle,
                    "final_attack_prompt": final.attack_prompt,
                    "final_jailbroken": final.jailbroken,
                    "final_judge_reason": final.judge_reason,
                }
            )
            final_csvfile.flush()

            status = "SUCCESS" if final.jailbroken else "failed"
            print(f"  → {status} after {len(records)} cycle(s)\n")

    print(f"Done. Results saved to {args.output}")
    print(f"Final prompts saved to {args.final_prompts_output}")


if __name__ == "__main__":
    main()
