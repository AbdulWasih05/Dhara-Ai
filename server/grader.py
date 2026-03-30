# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic grading functions for the Dhara AI environment.

Every function here is pure and deterministic: identical inputs always
produce identical outputs. No randomness, no LLM calls, no semantic
similarity models. Scoring is done exclusively via:

    - case-insensitive keyword / phrase substring matching
    - forbidden-keyword hard constraints
    - tier-based scoring against the grading_rules defined on each
      issue clause in server/scenarios.py

Scoring model
-------------
Every scenario's ScoringRubric stores the per-action point values
(e.g. NDA flag_issue_reward = 0.12). Component graders sum those values
over the agent's state and return the component contribution directly.
The final score is simply the sum of all component contributions,
clamped to [0.0, 1.0].

Max possible component scores (by design, all sum to exactly 1.0):

    NDA:  0.36 + 0.45 + 0.10 + 0.09                      = 1.00
    SaaS: 0.32 + 0.40 + 0.15 + 0.06 + 0.07               = 1.00
    JV:   0.30 + 0.36 + 0.18 + 0.04 + 0.06 + 0.06        = 1.00
"""

from typing import Dict, Iterable, List

try:
    from .scenarios import Clause, Scenario
except ImportError:  # pragma: no cover -- fallback for non-package execution
    from scenarios import Clause, Scenario


# ---------------------------------------------------------------------------
# Primitive: single-revision quality score
# ---------------------------------------------------------------------------


def grade_revision_quality(clause: Clause, revised_text: str) -> float:
    """Score a single revision's quality against one clause in [0.0, 1.0].

    Algorithm (fully deterministic):
      1. If the clause has no issue or no grading_rules, return 0.0.
      2. If ``revised_text`` is empty, return 0.0.
      3. If any forbidden_keyword is present, return ``forbidden_score``.
      4. For each concept in ``grading_rules["concepts"]``, mark it as
         matched if any of its keyword alternatives appears (case-
         insensitive substring).
      5. Walk ``grading_rules["tiers"]`` in order. Return the first
         tier's score where every name in ``required`` is in the matched
         set AND ``len(matched) >= min_count``.
      6. If no tier matches, return 0.0.
    """
    if not clause.has_issue or not clause.grading_rules:
        return 0.0
    if not revised_text:
        return 0.0

    rules = clause.grading_rules
    text_lower = revised_text.lower()

    # 3. Forbidden keywords -- hard constraint.
    for word in rules.get("forbidden_keywords", []):
        if word.lower() in text_lower:
            return float(rules.get("forbidden_score", 0.0))

    # 4. Match concepts.
    matched: set = set()
    for concept_name, keywords in rules.get("concepts", {}).items():
        for kw in keywords:
            if kw.lower() in text_lower:
                matched.add(concept_name)
                break

    # 5. Walk tiers in declared order.
    for tier in rules.get("tiers", []):
        required = tier.get("required", [])
        min_count = tier.get("min_count", 0)
        required_ok = all(c in matched for c in required)
        count_ok = len(matched) >= min_count
        if required_ok and count_ok:
            return float(tier["score"])

    # 6. No tier matched.
    return 0.0


# ---------------------------------------------------------------------------
# Component graders
# ---------------------------------------------------------------------------


def grade_issue_identification(
    scenario: Scenario,
    issues_found: Iterable[str],
    false_flags: Iterable[str],
) -> float:
    """Issue-identification component contribution.

        = n_correct_flags * flag_issue_reward
        + n_false_flags   * false_flag_penalty   (penalty is negative)

    Arguments are iterables of clause_ids; duplicates are ignored via
    set coercion so a buggy state counter can never double-count.
    """
    r = scenario.rubric
    n_correct = len(set(issues_found))
    n_false = len(set(false_flags))
    return n_correct * r.flag_issue_reward + n_false * r.false_flag_penalty


def grade_revision_quality_component(
    scenario: Scenario,
    revisions: Dict[str, List[str]],
) -> float:
    """Revision-quality component contribution (summed across clauses).

    For each issue clause that has at least one revision, scores the
    BEST (max) quality across all of that clause's revisions and
    multiplies by ``scenario.rubric.revision_weight``. This rewards
    good-faith iteration rather than punishing it.

    Non-issue clauses are ignored (grade_revision_quality would return
    0.0 anyway).
    """
    weight = scenario.rubric.revision_weight
    clause_map = {c.id: c for c in scenario.clauses}
    total = 0.0

    for clause_id, texts in revisions.items():
        clause = clause_map.get(clause_id)
        if clause is None or not clause.has_issue or not texts:
            continue
        best_quality = max(
            grade_revision_quality(clause, t) for t in texts
        )
        total += best_quality * weight
    return total


def grade_negotiation(
    scenario: Scenario,
    negotiation_decisions: Dict[str, List[str]],
) -> float:
    """Negotiation-skill component contribution.

    For each counterparty rule in the scenario, look up the agent's
    decision on that clause's first counter-offer and compare against
    ``rule.correct_agent_response``. Scoring:

        correct decision -> + negotiation_correct_reward
        wrong decision   -> + negotiation_wrong_penalty   (negative)
        no decision      -> 0 (neither reward nor penalty)

    Only the first decision per clause is scored, matching the
    counterparty rules which define a single pushback per clause.
    """
    r = scenario.rubric
    score = 0.0
    for rule in scenario.counterparty_rules:
        decisions = negotiation_decisions.get(rule.clause_id, [])
        if not decisions:
            continue
        agent_decision = decisions[0]
        if agent_decision == rule.correct_agent_response:
            score += r.negotiation_correct_reward
        else:
            score += r.negotiation_wrong_penalty
    return score


def grade_clean_clause_handling(
    scenario: Scenario,
    correctly_accepted: Iterable[str],
    wrongly_flagged: Iterable[str],
) -> float:
    """Clean-clause-handling component contribution.

        = n_correctly_accepted * accept_clean_reward
        + n_wrongly_flagged    * wrong_flag_clean_penalty   (negative)

    NOTE: ``wrongly_flagged`` is the same set of clause_ids as
    ``false_flags`` passed to grade_issue_identification -- wrongly
    flagging a clean clause is penalized in BOTH components by design.
    """
    r = scenario.rubric
    n_accepted = len(set(correctly_accepted))
    n_wrong = len(set(wrongly_flagged))
    return (
        n_accepted * r.accept_clean_reward
        + n_wrong * r.wrong_flag_clean_penalty
    )


def grade_completeness(
    scenario: Scenario,
    clauses_addressed: int,
    total_clauses: int,
) -> float:
    """Completeness bonus: all-or-nothing.

    Returns ``scenario.rubric.completeness_bonus`` if every clause has
    been addressed (accepted or flagged+revised), otherwise 0.0.
    """
    if total_clauses <= 0:
        return 0.0
    if clauses_addressed >= total_clauses:
        return scenario.rubric.completeness_bonus
    return 0.0


def grade_cross_clause(
    scenario: Scenario,
    issues_found: Iterable[str],
) -> float:
    """Cross-clause reasoning bonus (applies only where rubric > 0.0).

    Only the JV scenario has ``cross_clause_bonus > 0``. Detecting
    cross-clause reasoning deterministically from clause_ids alone is
    inherently a proxy, so we award credit for flagging known
    interaction pairs. Each pair contributes half the configured bonus.

    JV interaction pairs:
      * (clause_3 Deadlock, clause_6 Board)
            -- board ties invoke the deadlock resolution mechanism
      * (clause_3 Deadlock, clause_8 Exit)
            -- both rely on a shared independent-valuer / FMV mechanism

    Correctly flagging all three of clauses 3, 6, 8 awards the full
    bonus (0.06 for JV). Flagging just clause 3 paired with either 6
    OR 8 awards half (0.03).
    """
    r = scenario.rubric
    if r.cross_clause_bonus <= 0.0:
        return 0.0

    found = set(issues_found)
    half = r.cross_clause_bonus / 2.0
    score = 0.0

    # Pair 1: Deadlock + Board
    if "clause_3" in found and "clause_6" in found:
        score += half
    # Pair 2: Deadlock + Exit
    if "clause_3" in found and "clause_8" in found:
        score += half

    return score


# ---------------------------------------------------------------------------
# Final score aggregator
# ---------------------------------------------------------------------------

# Canonical component keys -- the environment should populate these
# when calling compute_final_score. Extra keys are allowed and summed.
FINAL_SCORE_COMPONENTS = (
    "issue_identification",
    "revision_quality",
    "negotiation",
    "clean_clause_handling",
    "completeness",
    "cross_clause",
)


def compute_final_score(
    scenario: Scenario,
    component_scores: Dict[str, float],
) -> float:
    """Sum all component contributions and clamp to [0.0, 1.0].

    The ScoringRubric values are absolute points (not ratios), so no
    additional weighting is needed here -- each component's
    contribution already reflects its weight in the overall rubric.

    Example:
        compute_final_score(scenario, {
            "issue_identification":   0.36,
            "revision_quality":       0.45,
            "negotiation":            0.00,
            "clean_clause_handling":  0.10,
            "completeness":           0.09,
            "cross_clause":           0.00,
        })  # -> 1.0

    Unknown / extra keys are summed in as well, which makes this safe
    for future component additions. The ``scenario`` argument is kept
    in the signature for symmetry with other graders and in case future
    rubrics need scenario-dependent aggregation.
    """
    # Reference scenario to avoid linter complaints; keeps the signature
    # stable for callers even when aggregation is currently
    # scenario-independent.
    _ = scenario
    total = sum(float(v) for v in component_scores.values())
    return max(0.0, min(1.0, total))
