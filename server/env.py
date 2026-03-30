# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DharaAiEnvironment -- the core OpenEnv Environment for contract review.

This module implements reset(), step(), and state() semantics described
in dhara_ai_design.md and CLAUDE.md:

    reset(task_name)  -- load a scenario (nda | saas | jv), set every
                         clause to 'unreviewed', reset trackers, return
                         the initial ContractObservation.

    step(action)      -- dispatch on action_type, mutate clause state,
                         award a per-step reward. Every action consumes
                         exactly one step. If the agent hits max_steps
                         the environment auto-submits.

    state             -- full ContractState (episode bookkeeping).

Reward model
------------
Per-step rewards accumulate into state.accumulated_reward. The final
episode score is clamp(accumulated_reward, 0.0, 1.0). Per-action
rewards use the point values stored in ScoringRubric (scenarios.py):

    flag_issue on issue clause   : + flag_issue_reward
    flag_issue on clean clause   : + false_flag_penalty
                                   + wrong_flag_clean_penalty
                                   (both negative; design intentionally
                                    double-penalizes false flags)
    suggest_revision             : + delta * revision_weight
                                   (delta = max(0, new_quality - best_so_far))
    accept_clause (clean)        : + accept_clean_reward
    accept_clause (issue, missed): 0.0 (opportunity cost only)
    negotiation decision         : + negotiation_correct_reward  OR
                                   + negotiation_wrong_penalty
    submit_review / auto-submit  : + completeness_bonus (if all addressed)
                                   + cross_clause_bonus (JV only)

The delta-based revision reward ensures the total revision reward for
a clause equals max_quality * revision_weight regardless of how many
revision attempts the agent makes. This rewards iteration without
allowing reward farming.

Invalid actions (unknown clause_id, wrong clause state, missing
required fields) return reward=0.0 with an explanatory
``last_action_error`` -- the step still counts against the budget, per
the design-doc rule "each action consumes one step".
"""

from typing import Any, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from .grader import (
        grade_completeness,
        grade_cross_clause,
        grade_revision_quality,
    )
    from .models import (
        ClauseInfo,
        ClauseState,
        ContractAction,
        ContractObservation,
        ContractState,
    )
    from .scenarios import Clause, CounterpartyRule, Scenario, get_scenario
except ImportError:  # pragma: no cover -- non-package execution fallback
    from grader import (
        grade_completeness,
        grade_cross_clause,
        grade_revision_quality,
    )
    from models import (
        ClauseInfo,
        ClauseState,
        ContractAction,
        ContractObservation,
        ContractState,
    )
    from scenarios import Clause, CounterpartyRule, Scenario, get_scenario


_ADDRESSED_STATUSES = frozenset({"accepted", "revised"})
_DEFAULT_TASK = "nda"


class DharaAiEnvironment(Environment):
    """OpenEnv Environment for legal contract clause review and negotiation.

    The agent reviews a contract on behalf of a client, flags problematic
    clauses, proposes revisions, accepts clean clauses, and negotiates
    with a scripted counterparty. Each instance runs one episode at a
    time; create a new instance per concurrent session.
    """

    # Each instance owns its scenario + state exclusively.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, default_task: str = _DEFAULT_TASK) -> None:
        super().__init__()
        self._default_task = default_task
        self._scenario: Optional[Scenario] = None
        self._state: ContractState = ContractState()
        # Internal tracker for delta-based revision rewards. Not stored
        # on ContractState because it is private bookkeeping.
        self._best_revision_quality: dict = {}

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ContractObservation:
        """Begin a new episode.

        ``task_name`` may be passed via kwargs (e.g. from the HTTP
        /reset body). Accepts "nda", "saas", or "jv". If omitted or
        unknown, falls back to the instance's default_task and surfaces
        the fallback via ``last_action_error``.
        """
        del seed  # Graders and counterparty responses are deterministic.

        requested = kwargs.get("task_name", None) or self._default_task
        warning: Optional[str] = None
        try:
            scenario = get_scenario(requested)
            task_name = requested
        except KeyError:
            scenario = get_scenario(self._default_task)
            task_name = self._default_task
            warning = (
                f"Unknown task_name {requested!r}; falling back to "
                f"{self._default_task!r}. Valid tasks: nda, saas, jv."
            )

        self._scenario = scenario
        self._best_revision_quality = {}

        clauses = {
            c.id: ClauseState(clause_id=c.id, status="unreviewed")
            for c in scenario.clauses
        }

        self._state = ContractState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            clauses=clauses,
            issues_found=[],
            false_flags=[],
            revisions={},
            negotiation_rounds={},
            negotiation_decisions={},
            max_steps=scenario.max_steps,
            accumulated_reward=0.0,
            episode_done=False,
        )

        result_msg = (
            f"Episode started. task={task_name}, "
            f"clauses={len(scenario.clauses)}, max_steps={scenario.max_steps}. "
            f"Review every clause and call submit_review when done."
        )
        return self._build_observation(
            reward=0.0,
            last_action_result=result_msg,
            last_action_error=warning,
        )

    # ------------------------------------------------------------------- step

    def step(
        self,
        action: ContractAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ContractObservation:
        """Execute a single agent action and return the next observation."""
        del timeout_s, kwargs

        if self._scenario is None:
            return self._build_observation(
                reward=0.0,
                last_action_error=(
                    "Environment has not been reset. Call reset() first."
                ),
            )

        if self._state.episode_done:
            return self._build_observation(
                reward=0.0,
                last_action_error=(
                    "Episode is already finished. Call reset() to start a new one."
                ),
            )

        # Every action consumes one step, even malformed ones.
        self._state.step_count += 1

        handlers = {
            "flag_issue": self._handle_flag_issue,
            "suggest_revision": self._handle_suggest_revision,
            "accept_clause": self._handle_accept_clause,
            "request_clarification": self._handle_request_clarification,
            "submit_review": self._handle_submit_review,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            # Should be unreachable because Literal validates at pydantic
            # layer, but we guard defensively.
            return self._build_observation(
                reward=0.0,
                last_action_error=f"Unknown action_type: {action.action_type!r}",
            )

        reward, result, error = handler(action)

        # Only accumulate reward on successful actions.
        if error is None:
            self._state.accumulated_reward += reward

        # Auto-submit if we've run out of steps without explicit submit.
        if (
            not self._state.episode_done
            and self._state.step_count >= self._state.max_steps
        ):
            bonus_reward, bonus_msg = self._finalize_episode(auto=True)
            reward += bonus_reward
            self._state.accumulated_reward += bonus_reward
            result = (
                (result + " | " if result else "")
                + f"[AUTO-SUBMIT] {bonus_msg}"
            )

        return self._build_observation(
            reward=reward,
            last_action_result=result,
            last_action_error=error,
        )

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> ContractState:
        """Return the current internal state snapshot."""
        return self._state

    # ------------------------------------------------------------- metadata

    def get_metadata(self):  # type: ignore[override]
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="dhara_ai",
            description=(
                "Legal contract clause-review and negotiation environment. "
                "Agent reviews clauses on behalf of a client, flags issues, "
                "proposes revisions, and negotiates with a scripted "
                "counterparty. Three tasks: nda (easy), saas (medium), "
                "jv (hard)."
            ),
            version="0.1.0",
        )

    # ========================================================== handlers

    def _handle_flag_issue(
        self, action: ContractAction
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if not action.clause_id:
            return 0.0, None, "flag_issue requires 'clause_id'."
        if not action.issue:
            return 0.0, None, "flag_issue requires a non-empty 'issue' description."

        scen_clause = self._get_scenario_clause(action.clause_id)
        if scen_clause is None:
            return 0.0, None, f"Unknown clause_id: {action.clause_id!r}."

        clause_state = self._state.clauses[action.clause_id]
        if clause_state.status != "unreviewed":
            return (
                0.0,
                None,
                (
                    f"Cannot flag {action.clause_id}: clause is already "
                    f"'{clause_state.status}'. Only 'unreviewed' clauses "
                    f"can be flagged."
                ),
            )

        clause_state.status = "flagged"
        clause_state.flag_issue_text = action.issue
        rubric = self._scenario.rubric  # type: ignore[union-attr]

        if scen_clause.has_issue:
            if action.clause_id not in self._state.issues_found:
                self._state.issues_found.append(action.clause_id)
            reward = rubric.flag_issue_reward
            result = (
                f"Flagged {action.clause_id}. Correct -- this clause has "
                f"a real issue. Reward={reward:+.2f}. Next, consider "
                f"suggest_revision to propose a fix."
            )
        else:
            if action.clause_id not in self._state.false_flags:
                self._state.false_flags.append(action.clause_id)
            # Both components penalize a false flag (see design doc).
            reward = rubric.false_flag_penalty + rubric.wrong_flag_clean_penalty
            result = (
                f"Flagged {action.clause_id}. FALSE FLAG -- this clause "
                f"was actually clean. Penalty={reward:+.2f}."
            )

        return reward, result, None

    def _handle_suggest_revision(
        self, action: ContractAction
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if not action.clause_id:
            return 0.0, None, "suggest_revision requires 'clause_id'."
        if not action.revised_text:
            return (
                0.0,
                None,
                "suggest_revision requires non-empty 'revised_text'.",
            )

        scen_clause = self._get_scenario_clause(action.clause_id)
        if scen_clause is None:
            return 0.0, None, f"Unknown clause_id: {action.clause_id!r}."

        clause_state = self._state.clauses[action.clause_id]
        if clause_state.status not in ("flagged", "revised", "negotiating"):
            return (
                0.0,
                None,
                (
                    f"Cannot revise {action.clause_id}: clause is "
                    f"'{clause_state.status}'. Clause must be 'flagged', "
                    f"'revised', or 'negotiating'."
                ),
            )

        prior_status = clause_state.status

        # Record the revision on both internal trackers.
        clause_state.revisions.append(action.revised_text)
        self._state.revisions.setdefault(action.clause_id, []).append(
            action.revised_text
        )

        # Delta-reward on quality improvement only.
        quality = grade_revision_quality(scen_clause, action.revised_text)
        rubric = self._scenario.rubric  # type: ignore[union-attr]
        best_so_far = self._best_revision_quality.get(action.clause_id, 0.0)
        delta = max(0.0, quality - best_so_far)
        revision_reward = delta * rubric.revision_weight
        self._best_revision_quality[action.clause_id] = max(best_so_far, quality)

        negotiation_reward = 0.0
        extra = ""

        if prior_status == "negotiating":
            # Re-revising during negotiation = REJECT the counter-offer.
            decisions = self._state.negotiation_decisions.setdefault(
                action.clause_id, []
            )
            decisions.append("REJECT")
            cp_rule = self._get_counterparty_rule(action.clause_id)
            # Score only the FIRST decision per clause (matches grader).
            if cp_rule is not None and len(decisions) == 1:
                if cp_rule.correct_agent_response == "REJECT":
                    negotiation_reward = rubric.negotiation_correct_reward
                    extra = (
                        f" Rejected counter-offer (correct). "
                        f"Negotiation reward={negotiation_reward:+.2f}."
                    )
                else:
                    negotiation_reward = rubric.negotiation_wrong_penalty
                    extra = (
                        f" Rejected counter-offer (wrong; counterparty's "
                        f"offer was acceptable). Penalty={negotiation_reward:+.2f}."
                    )
            clause_state.negotiation_history.append(
                "[Agent] Rejected counter-offer and proposed new revision."
            )
            clause_state.status = "revised"
        elif prior_status == "flagged":
            # First revision on a flagged clause.
            clause_state.status = "revised"
            clause_state.negotiation_history.append("[Agent] Proposed revision.")
            # Scripted counterparty pushback, if configured.
            cp_rule = self._get_counterparty_rule(action.clause_id)
            if cp_rule is not None:
                clause_state.status = "negotiating"
                clause_state.negotiation_history.append(
                    f"[Counterparty] {cp_rule.response_text}"
                )
                self._state.negotiation_rounds.setdefault(
                    action.clause_id, []
                ).append(cp_rule.response_text)
                extra = f" Counterparty pushback: '{cp_rule.response_text}'"
        else:
            # prior_status == "revised": iterative re-revision without
            # counterparty involvement. Status stays "revised". Delta
            # reward still applies if the new revision improves quality.
            clause_state.negotiation_history.append(
                "[Agent] Proposed updated revision."
            )

        total_reward = revision_reward + negotiation_reward
        result = (
            f"Revision recorded for {action.clause_id}. "
            f"quality={quality:.2f}, revision_reward={revision_reward:+.2f}."
            + extra
        )
        return total_reward, result, None

    def _handle_accept_clause(
        self, action: ContractAction
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if not action.clause_id:
            return 0.0, None, "accept_clause requires 'clause_id'."

        scen_clause = self._get_scenario_clause(action.clause_id)
        if scen_clause is None:
            return 0.0, None, f"Unknown clause_id: {action.clause_id!r}."

        clause_state = self._state.clauses[action.clause_id]
        status = clause_state.status
        if status not in ("unreviewed", "negotiating"):
            return (
                0.0,
                None,
                (
                    f"Cannot accept {action.clause_id}: clause is "
                    f"'{status}'. Must be 'unreviewed' or 'negotiating'."
                ),
            )

        rubric = self._scenario.rubric  # type: ignore[union-attr]

        if status == "unreviewed":
            clause_state.status = "accepted"
            if scen_clause.has_issue:
                # Missed the issue -- opportunity cost, no direct penalty.
                return (
                    0.0,
                    (
                        f"Accepted {action.clause_id} (unreviewed). "
                        f"Note: this clause has a real issue that was "
                        f"missed. No penalty, but no reward."
                    ),
                    None,
                )
            reward = rubric.accept_clean_reward
            return (
                reward,
                (
                    f"Accepted {action.clause_id}. Clean clause correctly "
                    f"accepted. Reward={reward:+.2f}."
                ),
                None,
            )

        # status == "negotiating" -> accepting the counter-offer
        decisions = self._state.negotiation_decisions.setdefault(
            action.clause_id, []
        )
        decisions.append("ACCEPT")
        cp_rule = self._get_counterparty_rule(action.clause_id)
        reward = 0.0
        extra = ""
        if cp_rule is not None and len(decisions) == 1:
            if cp_rule.correct_agent_response == "ACCEPT":
                reward = rubric.negotiation_correct_reward
                extra = (
                    f" (Accepting was correct -- counter-offer was "
                    f"reasonable.) reward={reward:+.2f}"
                )
            else:
                reward = rubric.negotiation_wrong_penalty
                extra = (
                    f" (Accepting was WRONG -- counter-offer violated "
                    f"client's hard limits.) penalty={reward:+.2f}"
                )
        clause_state.negotiation_history.append("[Agent] Accepted counter-offer.")
        clause_state.status = "accepted"
        return (
            reward,
            f"Accepted counter-offer on {action.clause_id}.{extra}",
            None,
        )

    def _handle_request_clarification(
        self, action: ContractAction
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if not action.clause_id:
            return 0.0, None, "request_clarification requires 'clause_id'."
        if not action.question:
            return (
                0.0,
                None,
                "request_clarification requires a non-empty 'question'.",
            )

        scen_clause = self._get_scenario_clause(action.clause_id)
        if scen_clause is None:
            return 0.0, None, f"Unknown clause_id: {action.clause_id!r}."

        if scen_clause.has_issue:
            clarification = (
                f"[Counsel -- clarification on {action.clause_id}] "
                f"Concern raised by the review desk: {scen_clause.issue_description}"
            )
        else:
            clarification = (
                f"[Counsel -- clarification on {action.clause_id}] "
                f"This clause appears standard and satisfies client "
                f"requirements; no concerns noted."
            )
        return 0.0, clarification, None

    def _handle_submit_review(
        self, action: ContractAction
    ) -> Tuple[float, Optional[str], Optional[str]]:
        del action  # submit_review takes no extra fields
        bonus_reward, bonus_msg = self._finalize_episode(auto=False)
        return bonus_reward, f"Review submitted. {bonus_msg}", None

    # ========================================================== internals

    def _finalize_episode(self, auto: bool) -> Tuple[float, str]:
        """Award terminal bonuses and mark the episode as done."""
        scenario = self._scenario
        assert scenario is not None  # guarded by callers

        total_clauses = len(scenario.clauses)
        addressed = sum(
            1
            for cs in self._state.clauses.values()
            if cs.status in _ADDRESSED_STATUSES
        )

        completeness = grade_completeness(
            scenario, clauses_addressed=addressed, total_clauses=total_clauses
        )
        cross = grade_cross_clause(scenario, issues_found=self._state.issues_found)
        total = completeness + cross
        self._state.episode_done = True

        final_score = max(
            0.0, min(1.0, self._state.accumulated_reward + total)
        )

        prefix = "Auto-submitted (step budget exhausted)." if auto else "Submitted."
        msg = (
            f"{prefix} Clauses addressed: {addressed}/{total_clauses}. "
            f"Completeness bonus={completeness:+.2f}, "
            f"cross-clause bonus={cross:+.2f}. "
            f"Final score={final_score:.4f}"
        )
        return total, msg

    def _get_scenario_clause(self, clause_id: str) -> Optional[Clause]:
        if self._scenario is None:
            return None
        for c in self._scenario.clauses:
            if c.id == clause_id:
                return c
        return None

    def _get_counterparty_rule(
        self, clause_id: str
    ) -> Optional[CounterpartyRule]:
        if self._scenario is None:
            return None
        for r in self._scenario.counterparty_rules:
            if r.clause_id == clause_id:
                return r
        return None

    def _build_observation(
        self,
        reward: float,
        last_action_result: Optional[str] = None,
        last_action_error: Optional[str] = None,
    ) -> ContractObservation:
        scenario = self._scenario
        if scenario is None:
            # Pre-reset: return a minimal observation.
            return ContractObservation(
                contract_type="",
                client_role="",
                client_requirements=[],
                clauses=[],
                steps_remaining=0,
                review_progress="no task loaded",
                last_action_result=last_action_result,
                last_action_error=(
                    last_action_error
                    or "Environment not initialized. Call reset()."
                ),
                done=False,
                reward=float(reward),
            )

        total_clauses = len(scenario.clauses)
        addressed = sum(
            1
            for cs in self._state.clauses.values()
            if cs.status in _ADDRESSED_STATUSES
        )
        clause_infos = [
            ClauseInfo(
                id=c.id,
                title=c.title,
                text=c.text,
                status=self._state.clauses[c.id].status,
                negotiation_history=list(
                    self._state.clauses[c.id].negotiation_history
                ),
            )
            for c in scenario.clauses
        ]
        steps_remaining = max(
            0, self._state.max_steps - self._state.step_count
        )

        return ContractObservation(
            contract_type=scenario.contract_type,
            client_role=scenario.client_role,
            client_requirements=list(scenario.client_requirements),
            clauses=clause_infos,
            steps_remaining=steps_remaining,
            review_progress=f"{addressed}/{total_clauses} clauses addressed",
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            done=self._state.episode_done,
            reward=float(reward),
        )
