"""Reward functions for GRPO training."""

from functools import lru_cache
import warnings

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

warnings.filterwarnings(
    "ignore",
    message="equations=True in NormalizationConfig is deprecated.*",
)

# `equations=True` was removed because math_verify now handles equations in the
# parser. Keeping it produces one warning per reward call, which is very costly
# inside GRPO rollouts.
_ANSWER_EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False,
            malformed_operators=False,
            basic_latex=True,
            boxed="all",
            units=True,
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
]


@lru_cache(maxsize=8192)
def _parse_gold_solution(solution: str):
    return parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )


def accuracy_reward_func(completions, solution, **kwargs):
    """Reward function that checks whether each completion matches its ground truth."""
    rewards = []
    for content, sol in zip(completions, solution):
        gold_parsed = _parse_gold_solution(str(sol))
        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                extraction_config=_ANSWER_EXTRACTION_CONFIG,
                extraction_mode="first_match",
            )
            try:
                reward = float(verify(answer_parsed, gold_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0
        else:
            reward = 1.0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward_func(completions, **kwargs):
    """Reward for enclosing reasoning in <think> tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    return [count_tags(c) for c in completions]
