"""Orchestrates the mutating attack loop for a single JBB behavior."""

from dataclasses import dataclass

from .attacker import generate_initial_attack, mutate_attack
from .config import FINAL_EVAL_ATTEMPTS, MAX_CYCLES
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


@dataclass
class FinalEvalRecord:
    behavior: str
    attempt: int
    attack_prompt: str
    target_response: str
    jailbroken: bool
    judge_reason: str


def _print_model_exchange(
    prefix: str,
    attack_prompt: str,
    target_response: str,
    judge_response: str,
) -> None:
    print(f"    [{prefix}] attack prompt:\n{attack_prompt}\n")
    print(f"    [{prefix}] target response:\n{target_response}\n")
    print(f"    [{prefix}] judge response:\n{judge_response}\n")


def run_behavior(
    goal: str,
    behavior: str,
    model_api_key: str,
    openai_api_key: str,
    max_cycles: int = MAX_CYCLES,
    initial_attack_prompt: str | None = None,
) -> list[CycleRecord]:
    """Run the mutating attack loop for one behavior.

    Returns one CycleRecord per cycle attempted, stopping early on jailbreak.
    Errors in any single model call are caught, logged as empty strings,
    and the loop continues so one bad API response doesn't abort the run.
    """
    records: list[CycleRecord] = []

    # Seed cycle 1 either from a provided jailbreak prompt (e.g., JBB artifact)
    # or by generating a fresh attack from the attacker model.
    if initial_attack_prompt is not None:
        attack_prompt = initial_attack_prompt
    else:
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
            judge_response = result.raw_response
        except Exception as exc:
            print(f"[judge error] {exc}")
            jailbroken = False
            judge_reason = f"judge error: {exc}"
            judge_response = judge_reason

        print("JAILBROKEN" if jailbroken else "not jailbroken")
        _print_model_exchange(
            prefix=f"cycle {cycle}",
            attack_prompt=attack_prompt,
            target_response=target_response,
            judge_response=judge_response,
        )

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
            try:
                attack_prompt = mutate_attack(goal, attack_prompt, feedback, model_api_key)
            except Exception as exc:
                print(f"  [attacker mutation error] {exc}")

    return records


def evaluate_final_prompt(
    goal: str,
    behavior: str,
    attack_prompt: str,
    model_api_key: str,
    openai_api_key: str,
    attempts: int = FINAL_EVAL_ATTEMPTS,
) -> list[FinalEvalRecord]:
    """Evaluate the final attack prompt multiple times and return pass/fail records."""
    records: list[FinalEvalRecord] = []

    for attempt in range(1, attempts + 1):
        print(f"    final eval attempt {attempt}/{attempts} …", end=" ", flush=True)

        try:
            target_response = query_target(attack_prompt, model_api_key)
        except Exception as exc:
            print(f"[target error] {exc}")
            target_response = ""

        try:
            result = judge(goal, attack_prompt, target_response, openai_api_key)
            jailbroken = result.jailbroken
            judge_reason = result.reason
            judge_response = result.raw_response
        except Exception as exc:
            print(f"[judge error] {exc}")
            jailbroken = False
            judge_reason = f"judge error: {exc}"
            judge_response = judge_reason

        print("PASS" if jailbroken else "FAIL")
        _print_model_exchange(
            prefix=f"final eval {attempt}",
            attack_prompt=attack_prompt,
            target_response=target_response,
            judge_response=judge_response,
        )

        records.append(
            FinalEvalRecord(
                behavior=behavior,
                attempt=attempt,
                attack_prompt=attack_prompt,
                target_response=target_response,
                jailbroken=jailbroken,
                judge_reason=judge_reason,
            )
        )

    return records
