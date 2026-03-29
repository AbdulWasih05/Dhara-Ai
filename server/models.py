# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic models for the Dhara AI contract-negotiation environment.

The environment exposes three OpenEnv-compliant types:

    ContractAction       -- what the agent sends on each step()
    ContractObservation  -- what the agent receives after each step/reset
    ContractState        -- full internal episode state

The action type is a discriminated union over five verbs
(flag_issue, suggest_revision, accept_clause, request_clarification,
submit_review) -- see ActionType below. Per-action-type semantic
validation (e.g. "flag_issue requires clause_id and issue") is
deliberately NOT enforced here; the environment handles malformed
actions gracefully by populating ``last_action_error`` in the
observation rather than returning HTTP 422. Only the discriminator
(``action_type``) is strictly validated.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums (Literal types)
# ---------------------------------------------------------------------------

ClauseStatus = Literal[
    "unreviewed",
    "flagged",
    "revised",
    "accepted",
    "negotiating",
]
"""Clause lifecycle status.

State transitions (see dhara_ai_design.md):
    unreviewed  -> flagged     (flag_issue)
    unreviewed  -> accepted    (accept_clause)
    flagged     -> revised     (suggest_revision)
    revised     -> negotiating (counterparty pushes back)
    negotiating -> revised     (agent sends another revision)
    negotiating -> accepted    (agent accepts counter-offer)
"""

ActionType = Literal[
    "flag_issue",
    "suggest_revision",
    "accept_clause",
    "request_clarification",
    "submit_review",
]
"""The five action verbs the agent may take per step."""


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class ClauseInfo(BaseModel):
    """Public per-clause information exposed in observations.

    This is what the agent sees. It intentionally excludes grading
    keywords, ideal revisions, and any other ground-truth data that
    would let the agent cheat.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    id: str = Field(..., description="Stable clause identifier, e.g. 'clause_1'")
    title: str = Field(..., description="Short human-readable clause title")
    text: str = Field(..., description="The current clause text")
    status: ClauseStatus = Field(
        default="unreviewed",
        description="Current lifecycle status of this clause",
    )
    negotiation_history: List[str] = Field(
        default_factory=list,
        description=(
            "Running narrative of negotiation turns on this clause "
            "(agent revisions, counterparty responses). Visible to the agent."
        ),
    )


class ClauseState(BaseModel):
    """Internal per-clause state tracked by the environment.

    Kept separate from ClauseInfo so the environment can carry richer
    bookkeeping (the agent's flagged-issue text, revision history) and
    filter the public view.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    clause_id: str = Field(..., description="Which clause this state tracks")
    status: ClauseStatus = Field(
        default="unreviewed", description="Current lifecycle status"
    )
    flag_issue_text: Optional[str] = Field(
        default=None,
        description="The issue description the agent gave when flagging",
    )
    revisions: List[str] = Field(
        default_factory=list,
        description="All revision texts the agent has proposed, in order",
    )
    negotiation_history: List[str] = Field(
        default_factory=list,
        description="Human-readable narrative of negotiation turns",
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class ContractAction(Action):
    """A single agent action.

    Only ``action_type`` is strictly validated (Literal discriminator).
    All other fields are optional at the model level because the
    environment validates required-field combinations semantically and
    surfaces errors via ``last_action_error`` in the observation rather
    than rejecting the request at the HTTP layer.

    Required-field contract (enforced by env.py, not here):

        flag_issue             -> clause_id, issue
        suggest_revision       -> clause_id, revised_text
        accept_clause          -> clause_id
        request_clarification  -> clause_id, question
        submit_review          -> (no extra fields)
    """

    action_type: ActionType = Field(
        ..., description="Which of the five action verbs the agent is invoking"
    )
    clause_id: Optional[str] = Field(
        default=None,
        description="Target clause id. Required for all actions except submit_review.",
    )
    issue: Optional[str] = Field(
        default=None,
        description="Description of the problem. Required for flag_issue.",
    )
    revised_text: Optional[str] = Field(
        default=None,
        description="Proposed new clause text. Required for suggest_revision.",
    )
    question: Optional[str] = Field(
        default=None,
        description="Clarification question. Required for request_clarification.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class ContractObservation(Observation):
    """What the agent sees after each reset() / step().

    Inherits ``done``, ``reward``, and ``metadata`` from the OpenEnv
    Observation base class. All other fields are environment-specific.
    """

    contract_type: str = Field(
        default="",
        description="Contract category, e.g. 'Non-Disclosure Agreement'",
    )
    client_role: str = Field(
        default="",
        description="Role the agent is representing, e.g. 'Disclosing Party'",
    )
    client_requirements: List[str] = Field(
        default_factory=list,
        description="List of the client's stated requirements / hard limits",
    )
    clauses: List[ClauseInfo] = Field(
        default_factory=list,
        description="Current state of every clause in the contract",
    )
    steps_remaining: int = Field(
        default=0,
        ge=0,
        description="Steps left before the episode is force-terminated",
    )
    review_progress: str = Field(
        default="",
        description="Human-readable progress string, e.g. '2/5 clauses addressed'",
    )
    last_action_result: Optional[str] = Field(
        default=None,
        description=(
            "Feedback from the environment on the previous action "
            "(e.g. counterparty response text, confirmation message). "
            "None on the initial reset observation."
        ),
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description=(
            "Error message if the previous action was malformed or "
            "illegal given the current clause state. None on success."
        ),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ContractState(State):
    """Full internal episode state.

    Inherits ``episode_id`` and ``step_count`` from the OpenEnv State
    base class. ``step_count`` is the canonical step counter -- we do
    NOT add a separate ``current_step`` field.

    The base class sets ``extra='allow'``, which is preserved here.
    """

    task_name: Optional[str] = Field(
        default=None,
        description="Active scenario id: 'nda' | 'saas' | 'jv'",
    )
    clauses: Dict[str, ClauseState] = Field(
        default_factory=dict,
        description="Per-clause internal state, keyed by clause_id",
    )
    issues_found: List[str] = Field(
        default_factory=list,
        description="Clause ids the agent correctly flagged as having an issue",
    )
    false_flags: List[str] = Field(
        default_factory=list,
        description="Clause ids the agent wrongly flagged (were actually clean)",
    )
    revisions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Per-clause list of proposed revision texts (in order)",
    )
    negotiation_rounds: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Per-clause list of counterparty counter-offers issued",
    )
    negotiation_decisions: Dict[str, List[str]] = Field(
        default_factory=dict,
        description=(
            "Per-clause list of agent decisions on counter-offers "
            "('ACCEPT' or 'REJECT'), parallel to negotiation_rounds"
        ),
    )
    max_steps: int = Field(
        default=0,
        ge=0,
        description="Step budget for this episode (15 | 25 | 35)",
    )
    accumulated_reward: float = Field(
        default=0.0,
        description="Sum of per-step rewards so far",
    )
    episode_done: bool = Field(
        default=False,
        description="Whether the episode has terminated (submit_review or out of steps)",
    )
