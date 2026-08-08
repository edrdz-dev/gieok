"""Exact statistical tests, implemented here to keep the benchmarks dependency-free.

Sample sizes in this suite are small (tens of questions), which rules out the normal
approximations most libraries reach for by default. Everything below is exact.
"""

from dataclasses import dataclass
from math import comb, sqrt


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """Outcome of a paired comparison between two systems."""

    wins: int
    """Questions the first system answered and the second did not."""
    losses: int
    """Questions the second system answered and the first did not."""
    p_value: float

    @property
    def significant(self) -> bool:
        """Whether the difference clears the conventional 5% threshold."""
        return self.p_value < 0.05


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a proportion.

    Preferred over the textbook normal interval because it stays inside [0, 1] and
    behaves sensibly at 0% and 100%, which is exactly where small benchmarks live.

    Args:
        successes: Number of successes observed.
        trials: Number of trials.
        z: Standard score for the desired confidence, 1.96 for 95%.

    Returns:
        Lower and upper bounds of the interval.
    """
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def mcnemar_exact(first: list[bool], second: list[bool]) -> McNemarResult:
    """Compare two systems evaluated on the *same* questions.

    The paired test is the right one here and is far more powerful than treating the two
    result sets as independent samples: questions both systems get right, or both get
    wrong, carry no information about which is better, so only the disagreements count.

    Args:
        first: Per-question success flags for one system.
        second: Per-question success flags for the other, in the same order.

    Returns:
        Wins, losses and the two-sided exact p-value.

    Raises:
        ValueError: If the two result lists have different lengths.
    """
    if len(first) != len(second):
        raise ValueError(f"paired inputs must match: {len(first)} vs {len(second)}")

    wins = sum(1 for a, b in zip(first, second, strict=True) if a and not b)
    losses = sum(1 for a, b in zip(first, second, strict=True) if b and not a)
    discordant = wins + losses
    if discordant == 0:
        return McNemarResult(wins=0, losses=0, p_value=1.0)

    # Under the null the discordant pairs split like a fair coin; sum the tail and double.
    tail = sum(comb(discordant, i) for i in range(min(wins, losses) + 1))
    p = min(1.0, 2 * tail / 2**discordant)
    return McNemarResult(wins=wins, losses=losses, p_value=p)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Return the two-sided p-value for the 2x2 table ``[[a, b], [c, d]]``.

    For *unpaired* comparisons. Prefer :func:`mcnemar_exact` whenever both systems were
    evaluated on the same questions, which in this suite is always.

    Args:
        a: Successes of the first system.
        b: Failures of the first system.
        c: Successes of the second system.
        d: Failures of the second system.

    Returns:
        The two-sided p-value.
    """
    total = a + b + c + d
    row, col = a + b, a + c

    def probability(x: int) -> float:
        return comb(row, x) * comb(c + d, col - x) / comb(total, col)

    observed = probability(a)
    low, high = max(0, col - (c + d)), min(row, col)
    # Sum every table at least as extreme as the one observed; the epsilon absorbs the
    # float noise that would otherwise drop a table of nominally equal probability.
    return min(
        1.0,
        sum(
            probability(x) for x in range(low, high + 1) if probability(x) <= observed * (1 + 1e-9)
        ),
    )
