"""Orchestrates the mutating attack loop for a single JBB behavior."""

from dataclasses import dataclass

from .attacker import generate_initial_attack, mutate_attack
from .config import MAX_CYCLES
from .feedback import get_feedback
from .judge import judge
from .target import query_target


@dataclass
class CycleRecord:
    behavior: str
    cycle: int
    attack_prompt: str
    target_response: str
    jailbroken: bool
    judge_reason: str
    feedback: str  # empty string when jailbroken or on the final cycle


def run_behavior(
    goal: str,
    behavior: str,
    model_api_key: str,
    openai_api_key: str,
    max_cycles: int = MAX_CYCLES,
) -> list[CycleRecord]:
    """Run the mutating attack loop for one behavior.

    Returns one CycleRecord per cycle attempted (always all max_cycles).
    Errors in any single model call are caught, logged as empty strings,
    and the loop continues so one bad API response doesn't abort the run.
    """
    records: list[CycleRecord] = []

    try:
        attack_prompt = generate_initial_attack(goal, model_api_key)
    except Exception as exc:
        print(f"  [attacker error on initial generation] {exc}")
        attack_prompt = goal

    for cycle in range(1, max_cycles + 1):
        print(f"  cycle {cycle}/{max_cycles} …", end=" ", flush=True)

        try:
            target_response = query_target(attack_prompt, model_api_key)
        except Exception as exc:
            print(f"[target error] {exc}")
            target_response = ""

        try:
            result = judge(goal, attack_prompt, target_response, openai_api_key)
            jailbroken = result.jailbroken
            judge_reason = result.reason
        except Exception as exc:
            print(f"[judge error] {exc}")
            jailbroken = False
            judge_reason = f"judge error: {exc}"

        print("JAILBROKEN" if jailbroken else "not jailbroken")

        feedback = ""
        if not jailbroken and cycle < max_cycles:
            try:
                feedback = get_feedback(
                    goal, attack_prompt, target_response, judge_reason, openai_api_key
                )
            except Exception as exc:
                print(f"  [feedback error] {exc}")
                feedback = f"feedback error: {exc}"

        records.append(
            CycleRecord(
                behavior=behavior,
                cycle=cycle,
                attack_prompt=attack_prompt,
                target_response=target_response,
                jailbroken=jailbroken,
                judge_reason=judge_reason,
                feedback=feedback,
            )
        )

        if cycle < max_cycles:
            try:
                attack_prompt = mutate_attack(goal, attack_prompt, feedback, model_api_key)
            except Exception as exc:
                print(f"  [attacker mutation error] {exc}")

    return records
