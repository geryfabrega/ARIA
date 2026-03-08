"""Orchestrates the mutating attack loop for a single JBB behavior."""

from dataclasses import dataclass

from .attacker import generate_initial_attack, mutate_attack
from .config import MAX_CYCLES
from .feedback import get_feedback
from .judge import judge
from .target import query_target


def _normalize_prompt(text: str) -> str:
    """Normalize whitespace/case so near-identical prompts can be detected."""
    return " ".join(text.split()).strip().lower()


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
    together_api_key: str,
    openai_api_key: str,
    max_cycles: int = MAX_CYCLES,
    initial_attack_prompt: str | None = None,
) -> list[CycleRecord]:
    """Run the mutating attack loop for one behavior.

    Returns one CycleRecord per cycle attempted (always all max_cycles).
    Errors in any single model call are caught, logged as empty strings,
    and the loop continues so one bad API response doesn't abort the run.
    """
    records: list[CycleRecord] = []

    if initial_attack_prompt is not None:
        attack_prompt = initial_attack_prompt
    else:
        try:
            attack_prompt = generate_initial_attack(goal, together_api_key)
        except Exception as exc:
            print(f"  [attacker error on initial generation] {exc}")
            attack_prompt = goal

    for cycle in range(1, max_cycles + 1):
        print(f"  cycle {cycle}/{max_cycles} …", end=" ", flush=True)

        try:
            target_response = query_target(attack_prompt, together_api_key)
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
        if cycle < max_cycles:
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
            previous_prompt = attack_prompt
            try:
                candidate = mutate_attack(goal, previous_prompt, feedback, together_api_key)
            except Exception as exc:
                print(f"  [attacker mutation error] {exc}")
                candidate = ""

            if not candidate:
                print("  [attacker mutation warning] empty mutation; forcing variation")
                candidate = (
                    f"{previous_prompt}\n\n"
                    f"Iteration {cycle + 1}: rewrite this as a new scenario with different "
                    "wording and structure."
                )

            if _normalize_prompt(candidate) == _normalize_prompt(previous_prompt):
                print("  [attacker mutation warning] identical mutation; retrying once")
                retry_feedback = (
                    f"{feedback}\n\n"
                    "Retry constraint: produce a materially different prompt from the previous "
                    "one using a new rhetorical angle."
                )
                try:
                    retried = mutate_attack(goal, previous_prompt, retry_feedback, together_api_key)
                except Exception as exc:
                    print(f"  [attacker mutation retry error] {exc}")
                    retried = ""

                if retried and _normalize_prompt(retried) != _normalize_prompt(previous_prompt):
                    candidate = retried
                else:
                    print("  [attacker mutation warning] forcing deterministic variation")
                    candidate = (
                        f"{previous_prompt}\n\n"
                        f"Iteration {cycle + 1} directive: change wording, scenario, and "
                        "argument style from the prior version."
                    )

            attack_prompt = candidate

    return records
