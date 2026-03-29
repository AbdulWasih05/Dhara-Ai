# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario data for the Dhara AI environment.

All three contract negotiation scenarios are defined here as pure data
(dataclasses + dicts). No runtime logic lives in this module; the grader
(server/grader.py) and environment (server/env.py) consume these
structures to produce rewards and observations.

Scenarios:
    - nda:  Easy   -- Non-Disclosure Agreement (5 clauses, 3 issues, no negotiation)
    - saas: Medium -- SaaS Service Agreement  (7 clauses, 4 issues, light pushback)
    - jv:   Hard   -- Joint Venture Agreement (10 clauses, 6 issues, aggressive pushback)

The single source of truth for all contract content, grading keywords,
reward magnitudes, and counterparty rules is dhara_ai_design.md.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Clause:
    """A single clause within a contract scenario.

    For clauses where ``has_issue`` is True, ``issue_description``,
    ``ideal_revision``, and ``grading_rules`` are populated and consumed
    by the revision-quality grader. For clean clauses, those fields are
    left empty.
    """

    id: str
    title: str
    text: str
    has_issue: bool
    issue_description: str = ""
    ideal_revision: str = ""
    # Grading configuration for revision quality (only meaningful when
    # ``has_issue`` is True). See "Grading rules schema" below.
    grading_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterpartyRule:
    """A single counterparty pushback step attached to a revised clause.

    After the agent proposes a revision on ``clause_id``, the counterparty
    responds with ``response_text``. The agent is expected to either
    ACCEPT (via ``accept_clause``) or REJECT (via another
    ``suggest_revision``). The grader uses ``correct_agent_response`` to
    score the negotiation turn.
    """

    clause_id: str
    response_text: str
    correct_agent_response: str  # "ACCEPT" or "REJECT"
    reason: str = ""


@dataclass
class ScoringRubric:
    """Per-action reward magnitudes for a single scenario.

    These values combine to exactly 1.0 when an ideal agent hits every
    component perfectly. See dhara_ai_design.md for the derivation of
    each number.
    """

    flag_issue_reward: float          # per correct flag on an issue clause
    false_flag_penalty: float         # per flag on a clean clause (issue-id component)
    revision_weight: float            # per revision, multiplied by quality in [0, 1]
    accept_clean_reward: float        # per correct accept on a clean clause
    wrong_flag_clean_penalty: float   # per flag on a clean clause (clean-handling component)
    completeness_bonus: float         # paid once if every clause is addressed
    negotiation_correct_reward: float = 0.0  # per correct negotiation decision
    negotiation_wrong_penalty: float = 0.0   # per wrong negotiation decision
    cross_clause_bonus: float = 0.0          # paid once for cross-clause reasoning (JV only)


@dataclass
class Scenario:
    """A complete negotiation scenario (one task)."""

    task_id: str                       # "nda" | "saas" | "jv"
    contract_type: str
    client_name: str
    client_role: str
    client_requirements: List[str]
    max_steps: int
    clauses: List[Clause]
    counterparty_rules: List[CounterpartyRule]
    rubric: ScoringRubric


# ---------------------------------------------------------------------------
# Grading rules schema
# ---------------------------------------------------------------------------
#
# Each issue clause carries a ``grading_rules`` dict with this shape:
#
#   {
#       "concepts": {
#           "<concept_name>": ["keyword or phrase", ...],
#           ...
#       },
#       "forbidden_keywords": ["keyword", ...],   # hard constraint
#       "forbidden_score": 0.0,                   # score when any forbidden present
#       "tiers": [
#           # Ordered from highest to lowest score. First match wins.
#           # A tier matches when:
#           #   - every name in "required" is present in the matched set, AND
#           #   - len(matched set) >= min_count
#           # Either field may be omitted (defaults: required=[], min_count=0).
#           {"required": [...], "min_count": N, "score": 0.0},
#           ...
#       ],
#   }
#
# Keyword matching is case-insensitive substring matching (done in the
# grader). Concepts are satisfied when any of their keywords appear in
# the revised_text.
# ---------------------------------------------------------------------------


# ===========================================================================
# TASK 1 -- NDA (Easy)
# ===========================================================================

NDA_CLAUSES: List[Clause] = [
    Clause(
        id="clause_1",
        title="Definition of Confidential Information",
        text=(
            '"Confidential Information" shall mean any information explicitly '
            'marked as "CONFIDENTIAL" in writing by the Disclosing Party at '
            "the time of disclosure."
        ),
        has_issue=True,
        issue_description=(
            "Too narrow -- only covers information explicitly marked in "
            "writing. Verbal disclosures, source code shared via repos, and "
            "information obviously confidential by nature are all excluded. "
            "Client wants a broad definition."
        ),
        ideal_revision=(
            '"Confidential Information" shall mean any and all information '
            "disclosed by the Disclosing Party, whether orally, in writing, "
            "electronically, or by any other means, and whether or not "
            "marked as confidential, including but not limited to source "
            "code, algorithms, business plans, customer data, and any "
            "information that a reasonable person would consider "
            "confidential given the nature of the information and "
            "circumstances of disclosure."
        ),
        grading_rules={
            "concepts": {
                "oral_verbal": [
                    "oral", "orally", "verbal", "verbally",
                    "electronic", "electronically",
                    "any means", "any form", "any medium",
                ],
                "regardless_marking": [
                    "regardless", "whether or not", "without regard",
                    "not required to be marked", "irrespective of marking",
                    "not marked",
                ],
                "reasonable_person": [
                    "reasonable person", "reasonably consider",
                    "by nature", "nature of the information",
                    "nature of", "circumstances of disclosure",
                    "given the circumstances",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"min_count": 3, "score": 1.0},
                {"min_count": 2, "score": 0.7},
                {"min_count": 1, "score": 0.4},
                {"min_count": 0, "score": 0.1},
            ],
        },
    ),
    Clause(
        id="clause_2",
        title="Duration of Obligations",
        text=(
            "The obligations of confidentiality under this Agreement shall "
            "survive in perpetuity from the date of disclosure of each item "
            "of Confidential Information."
        ),
        has_issue=True,
        issue_description=(
            '"In perpetuity" violates the client\'s 3-year maximum '
            "requirement. Perpetual NDAs are also generally disfavored and "
            "difficult to enforce in Indian courts."
        ),
        ideal_revision=(
            "The obligations of confidentiality under this Agreement shall "
            "survive for a period of three (3) years from the date of "
            "disclosure of each item of Confidential Information."
        ),
        grading_rules={
            "concepts": {
                "finite_period": [
                    "year", "years", "month", "months", "period of",
                ],
                "within_limit": [
                    "three (3)", "3 year", "three year", "3-year",
                    "three-year", "two (2)", "2 year", "two year",
                    "2-year", "two-year", "one (1)", "1 year",
                    "one year", "1-year", "one-year",
                ],
            },
            "forbidden_keywords": [
                "perpetuity", "perpetual", "perpetually", "forever",
            ],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["finite_period", "within_limit"], "score": 1.0},
                {"required": ["finite_period"], "score": 0.5},
                {"min_count": 0, "score": 0.3},
            ],
        },
    ),
    Clause(
        id="clause_3",
        title="Return of Materials",
        text=(
            "Upon termination of this Agreement, the Receiving Party shall "
            "promptly destroy all copies of Confidential Information in any "
            "form, including electronic copies and backups, and shall "
            "provide written certification of such destruction within "
            "fourteen (14) days."
        ),
        has_issue=False,
    ),
    Clause(
        id="clause_4",
        title="Non-Solicitation",
        text=(
            "During the term of this Agreement and for a period of twelve "
            "(12) months thereafter, the Receiving Party shall not directly "
            "or indirectly solicit, recruit, or hire any employee or "
            "contractor of the Disclosing Party who was involved in the "
            "performance of obligations under this Agreement."
        ),
        has_issue=False,
    ),
    Clause(
        id="clause_5",
        title="Governing Law and Jurisdiction",
        text=(
            "This Agreement shall be governed by the laws of the State of "
            "Delaware, United States, and any disputes shall be subject to "
            "the exclusive jurisdiction of the courts located in Wilmington, "
            "Delaware."
        ),
        has_issue=True,
        issue_description=(
            "Client requires Indian jurisdiction (Bengaluru). Delaware "
            "jurisdiction is completely misaligned."
        ),
        ideal_revision=(
            "This Agreement shall be governed by the laws of India and any "
            "disputes arising under this Agreement shall be subject to the "
            "exclusive jurisdiction of the courts located in Bengaluru, "
            "Karnataka, India."
        ),
        grading_rules={
            "concepts": {
                "india_law": [
                    "laws of india", "indian law", "laws of the republic of india",
                    "india", "republic of india",
                ],
                "bengaluru": [
                    "bengaluru", "bangalore", "karnataka",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["india_law", "bengaluru"], "score": 1.0},
                {"required": ["india_law"], "score": 0.6},
                {"required": ["bengaluru"], "score": 0.5},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
]

NDA_SCENARIO = Scenario(
    task_id="nda",
    contract_type="Non-Disclosure Agreement",
    client_name="TechStart Innovations Pvt. Ltd.",
    client_role="Disclosing Party",
    client_requirements=[
        "Confidential information must be clearly and broadly defined.",
        "NDA duration should not exceed 3 years.",
        "Receiving party must destroy (not just return) materials on termination.",
        "Must include non-solicitation of employees.",
        "Jurisdiction must be Indian courts (Bengaluru).",
    ],
    max_steps=15,
    clauses=NDA_CLAUSES,
    counterparty_rules=[],  # Easy tier: no counterparty pushback.
    rubric=ScoringRubric(
        flag_issue_reward=0.12,
        false_flag_penalty=-0.04,
        revision_weight=0.15,
        accept_clean_reward=0.05,
        wrong_flag_clean_penalty=-0.03,
        completeness_bonus=0.09,
        negotiation_correct_reward=0.0,
        negotiation_wrong_penalty=0.0,
        cross_clause_bonus=0.0,
    ),
)


# ===========================================================================
# TASK 2 -- SaaS Service Agreement (Medium)
# ===========================================================================

SAAS_CLAUSES: List[Clause] = [
    Clause(
        id="clause_1",
        title="Limitation of Liability",
        text=(
            "Provider's total aggregate liability under this Agreement shall "
            "not exceed the total fees paid by Client in the twelve (12) "
            "months preceding the claim. In no event shall Provider be "
            "liable for any indirect, incidental, consequential, special, "
            "or punitive damages, including but not limited to loss of "
            "data, loss of revenue, or business interruption, regardless "
            "of the cause of action."
        ),
        has_issue=True,
        issue_description=(
            "The cap at 1x annual fees is acceptable (within client's 2x "
            "limit). However, excluding 'loss of data' from consequential "
            "damages is unacceptable for a healthtech company handling "
            "patient records. Data loss liability cannot be fully excluded."
        ),
        ideal_revision=(
            "Provider's total aggregate liability under this Agreement "
            "shall not exceed two (2) times the total fees paid by Client "
            "in the twelve (12) months preceding the claim. In no event "
            "shall Provider be liable for any indirect, incidental, "
            "consequential, special, or punitive damages, including but "
            "not limited to loss of revenue or business interruption, "
            "regardless of cause of action; provided however that this "
            "limitation shall not apply to Provider's obligations regarding "
            "data protection, data loss, or breach of confidentiality."
        ),
        grading_rules={
            "concepts": {
                "data_mention": [
                    "data protection", "data loss", "loss of data",
                    "breach of confidentiality", "data breach",
                    "patient data", "personal data",
                ],
                "carveout_language": [
                    "shall not apply", "except for", "excluding",
                    "provided however", "provided that", "carve-out",
                    "carveout", "notwithstanding",
                ],
                "cap_adjusted": [
                    "two (2) times", "2x", "2 times", "twice",
                    "two times", "two-times",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["data_mention", "carveout_language"], "score": 1.0},
                {"required": ["data_mention"], "score": 0.6},
                {"required": ["cap_adjusted"], "score": 0.3},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_2",
        title="Data Handling on Termination",
        text=(
            "Upon termination, Provider shall make Client's data available "
            "for download for a period of sixty (60) days. After this "
            "period, Provider reserves the right to delete or archive the "
            "data at its discretion."
        ),
        has_issue=True,
        issue_description=(
            "Two problems: (1) 60-day availability window but no guaranteed "
            "deletion deadline -- 'at its discretion' could mean indefinite "
            "retention. (2) Client requires deletion within 30 days, not "
            "60 days of optional access."
        ),
        ideal_revision=(
            "Upon termination, Provider shall make Client's data available "
            "for download for a period of thirty (30) days. At the end of "
            "this period, Provider shall permanently delete all Client "
            "data, including backups, within thirty (30) days and provide "
            "written certification of deletion. Provider shall not retain "
            "any copies of Client data beyond this period."
        ),
        grading_rules={
            "concepts": {
                "mandatory_deletion": [
                    "shall delete", "shall permanently delete",
                    "must delete", "permanently delete",
                    "shall not retain", "shall destroy",
                    "mandatory deletion",
                ],
                "thirty_days": [
                    "30 day", "thirty (30)", "thirty day", "30-day",
                    "thirty-day", "within 30", "within thirty",
                ],
                "certification": [
                    "certification", "certify", "written certification",
                    "proof of deletion", "certificate of destruction",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"min_count": 3, "score": 1.0},
                {"min_count": 2, "score": 0.6},
                {"min_count": 1, "score": 0.3},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_3",
        title="Service Level Agreement",
        text=(
            "Provider targets an uptime of 99.0% measured on a quarterly "
            "basis, excluding scheduled maintenance windows. Provider "
            "shall use commercially reasonable efforts to meet this "
            "target. No service credits or remedies shall apply for "
            "failure to meet the uptime target."
        ),
        has_issue=True,
        issue_description=(
            "Three problems: (1) 99.0% is below client's 99.5% requirement. "
            "(2) 'Targets' and 'commercially reasonable efforts' make the "
            "obligation non-binding. (3) No service credits."
        ),
        ideal_revision=(
            "Provider guarantees a minimum uptime of 99.5% measured on a "
            "monthly basis, excluding scheduled maintenance windows "
            "notified at least 72 hours in advance. For each 0.1% below "
            "the guaranteed uptime in any calendar month, Client shall "
            "receive a service credit equal to 5% of the monthly "
            "subscription fee, up to a maximum of 30% of the monthly fee. "
            "Service credits shall be applied to the next billing cycle "
            "automatically."
        ),
        grading_rules={
            "concepts": {
                "high_uptime": [
                    "99.5", "99.9", "99.99", "99.5%", "99.9%",
                ],
                "binding_guarantee": [
                    "guarantee", "guarantees", "guaranteed",
                    "shall maintain", "shall provide", "warrants",
                    "warranty", "commits to", "binding",
                ],
                "service_credits": [
                    "service credit", "service credits", "credit equal to",
                    "% credit", "monthly credit", "credit of",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"min_count": 3, "score": 1.0},
                {"min_count": 2, "score": 0.6},
                {"min_count": 1, "score": 0.3},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_4",
        title="Insurance",
        text=(
            "Provider shall maintain cyber liability insurance with "
            "coverage of not less than \u20B95 crore (INR 50,000,000) for "
            "the duration of this Agreement and shall provide proof of "
            "such insurance to Client upon request."
        ),
        has_issue=False,
    ),
    Clause(
        id="clause_5",
        title="Data Residency",
        text=(
            "Client data may be stored and processed in any of Provider's "
            "global data centers, including those located outside of "
            "India, to ensure optimal performance and redundancy."
        ),
        has_issue=True,
        issue_description=(
            "Directly violates data residency requirement. Patient health "
            "data stored outside India may also violate the DPDP Act and "
            "healthcare data regulations."
        ),
        ideal_revision=(
            "All Client data shall be stored and processed exclusively "
            "within data centers located in India. Provider shall not "
            "transfer, replicate, or process Client data outside of India "
            "without prior written consent from Client. Provider shall "
            "ensure compliance with applicable Indian data protection "
            "laws, including the Digital Personal Data Protection Act."
        ),
        grading_rules={
            "concepts": {
                "india_only": [
                    "exclusively within india", "exclusively in india",
                    "only within india", "only in india",
                    "solely within india", "solely in india",
                    "within india", "in india only",
                    "india-based", "located in india",
                ],
                "no_transfer": [
                    "shall not transfer", "not transfer",
                    "without prior written consent", "without consent",
                    "prior written consent", "not replicate",
                    "shall not replicate", "shall not process",
                ],
                "vague_india": [
                    "prefer india", "primarily india",
                    "where possible", "where feasible",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["india_only", "no_transfer"], "score": 1.0},
                {"required": ["india_only"], "score": 0.6},
                {"required": ["vague_india"], "score": 0.3},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_6",
        title="Data Usage Rights",
        text=(
            "Provider shall not use, analyze, or process Client data for "
            "any purpose other than providing the Services under this "
            "Agreement. Provider expressly shall not use Client data for "
            "training machine learning models, analytics products, or any "
            "form of data monetization."
        ),
        has_issue=False,
    ),
    Clause(
        id="clause_7",
        title="Termination",
        text=(
            "Either party may terminate this Agreement by providing "
            "ninety (90) days written notice to the other party. Upon "
            "early termination by Client, Provider shall refund a "
            "pro-rata portion of any prepaid fees for the unused portion "
            "of the subscription term within thirty (30) days of the "
            "termination effective date."
        ),
        has_issue=False,
    ),
]

SAAS_COUNTERPARTY_RULES: List[CounterpartyRule] = [
    CounterpartyRule(
        clause_id="clause_1",
        response_text=(
            "We accept the data protection carve-out, but the liability "
            "cap remains at 1x annual fees."
        ),
        correct_agent_response="ACCEPT",
        reason=(
            "1x cap is within client's 2x maximum limit; data carve-out "
            "secured -- take the win."
        ),
    ),
    CounterpartyRule(
        clause_id="clause_2",
        response_text=(
            "We can offer 45-day availability plus deletion, not 30."
        ),
        correct_agent_response="REJECT",
        reason=(
            "Client's deletion requirement is firm at 30 days -- do not "
            "concede on a hard limit."
        ),
    ),
    CounterpartyRule(
        clause_id="clause_3",
        response_text=(
            "We can offer 99.2% uptime with 2% service credits."
        ),
        correct_agent_response="REJECT",
        reason=(
            "Must push for 99.5% -- 99.2% does not meet client's SLA "
            "requirement."
        ),
    ),
    # NOTE: Clause 5 (Data Residency) is not included here. Per the design
    # doc, the counterparty accepts the revision immediately with no
    # pushback ("N/A"), so it is not a scored negotiation round. The
    # environment simply transitions the clause to "revised" without
    # requiring an agent response.
]

SAAS_SCENARIO = Scenario(
    task_id="saas",
    contract_type="SaaS Service Agreement",
    client_name="MediTrack Solutions",
    client_role="Buyer (Customer)",
    client_requirements=[
        "Liability cap must not exceed 2x annual subscription fees.",
        "All patient data must be deleted within 30 days of termination.",
        "Uptime SLA must be at least 99.5% with service credits for breaches.",
        "Provider must carry cyber insurance of at least \u20B95 crore.",
        "Data must remain within India (data residency).",
        "Provider cannot use client data for training ML models.",
        "90-day termination notice with pro-rata refund.",
    ],
    max_steps=25,
    clauses=SAAS_CLAUSES,
    counterparty_rules=SAAS_COUNTERPARTY_RULES,
    rubric=ScoringRubric(
        flag_issue_reward=0.08,
        false_flag_penalty=-0.03,
        revision_weight=0.10,
        accept_clean_reward=0.02,
        wrong_flag_clean_penalty=-0.02,
        completeness_bonus=0.07,
        negotiation_correct_reward=0.05,
        negotiation_wrong_penalty=-0.05,
        cross_clause_bonus=0.0,
    ),
)


# ===========================================================================
# TASK 3 -- Joint Venture Agreement (Hard)
# ===========================================================================

JV_CLAUSES: List[Clause] = [
    Clause(
        id="clause_1",
        title="Intellectual Property Ownership",
        text=(
            "All intellectual property, including patents, copyrights, "
            "trade secrets, and know-how, developed in the course of the "
            "Joint Venture shall be the exclusive property of the Joint "
            "Venture entity. Each Party's pre-existing intellectual "
            "property contributed to the Joint Venture shall be licensed "
            "to the JV entity on an irrevocable, royalty-free, perpetual, "
            "worldwide basis."
        ),
        has_issue=True,
        issue_description=(
            "Two problems: (1) JV-developed IP should be jointly owned by "
            "the parties, not owned exclusively by the JV entity (if the "
            "JV dissolves, client loses access). (2) Pre-existing IP "
            "licensed irrevocably and perpetually means the client loses "
            "control of their own IP forever."
        ),
        ideal_revision=(
            "All intellectual property developed in the course of the "
            "Joint Venture shall be jointly owned by the Parties in "
            "proportion to their respective stakes. Each Party's "
            "pre-existing intellectual property shall remain the exclusive "
            "property of the contributing Party and shall be licensed to "
            "the JV entity on a non-exclusive, revocable, royalty-free "
            "basis for the duration of the Joint Venture, terminating "
            "upon dissolution or exit of the contributing Party."
        ),
        grading_rules={
            "concepts": {
                "joint_ownership": [
                    "jointly owned", "joint ownership", "jointly held",
                    "proportion to their", "proportional to",
                    "owned by the parties", "by the parties",
                ],
                "revocable_license": [
                    "revocable", "non-exclusive",
                    "terminating upon", "terminate upon",
                    "terminating on dissolution", "duration of the joint venture",
                    "for the duration of the",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["joint_ownership", "revocable_license"], "score": 1.0},
                {"required": ["joint_ownership"], "score": 0.6},
                {"required": ["revocable_license"], "score": 0.4},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_2",
        title="Transfer of Stake / Right of First Refusal",
        text=(
            "Either Party may transfer its stake in the Joint Venture to "
            "any third party upon providing thirty (30) days written "
            "notice to the other Party. The non-transferring Party shall "
            "have no right to object to or block such transfer."
        ),
        has_issue=True,
        issue_description=(
            "No right of first refusal at all. Client could wake up with "
            "a hostile new partner without any recourse."
        ),
        ideal_revision=(
            "Before transferring any stake to a third party, the "
            "transferring Party must first offer its stake to the other "
            "Party on the same terms and conditions (Right of First "
            "Refusal). The non-transferring Party shall have sixty (60) "
            "days to exercise this right. Only upon written declination "
            "or expiry of this period may the transfer proceed to a "
            "third party."
        ),
        grading_rules={
            "concepts": {
                "rofr": [
                    "right of first refusal", "rofr",
                    "first offer", "first refusal",
                    "offer its stake to the other",
                    "offer the stake to", "must first offer",
                ],
                "timeline": [
                    "60 day", "sixty (60)", "sixty day", "60-day",
                    "30 day", "thirty (30)", "thirty day", "30-day",
                    "45 day", "forty-five (45)", "45-day",
                    "days to exercise", "days to respond",
                ],
                "some_restriction": [
                    "consent", "restrict", "approval",
                    "notify", "notice to",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["rofr", "timeline"], "score": 1.0},
                {"required": ["rofr"], "score": 0.6},
                {"required": ["some_restriction"], "score": 0.3},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_3",
        title="Deadlock Resolution",
        text=(
            "In the event of a deadlock in Board decisions, the Parties "
            "shall refer the matter to binding arbitration under the "
            "rules of the International Chamber of Commerce, with the "
            "seat of arbitration in New York."
        ),
        has_issue=True,
        issue_description=(
            "Two problems: (1) No buyout option as required by client -- "
            "only arbitration. (2) Seat of arbitration should be "
            "Singapore, not New York."
        ),
        ideal_revision=(
            "In the event of a deadlock persisting for more than thirty "
            "(30) days: (a) Either Party may initiate a buyout by "
            "offering to purchase the other Party's stake at fair market "
            "value determined by an independent valuer mutually agreed "
            "upon; (b) If no buyout is agreed within sixty (60) days, "
            "the matter shall be referred to binding arbitration under "
            "ICC rules with the seat of arbitration in Singapore."
        ),
        grading_rules={
            "concepts": {
                "buyout": [
                    "buyout", "buy-out", "buy out",
                    "offer to purchase", "purchase the other party",
                    "purchase the other's", "buy each other out",
                    "initiate a buyout", "buy the other",
                ],
                "singapore": [
                    "singapore", "siac",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["buyout", "singapore"], "score": 1.0},
                {"required": ["buyout"], "score": 0.6},
                {"required": ["singapore"], "score": 0.4},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_4",
        title="Non-Compete",
        text=(
            "Each Party agrees that during the term of the Joint Venture "
            "and in perpetuity thereafter, it shall not directly or "
            "indirectly engage in any business that competes with the "
            "Joint Venture's operations in any geography where the JV "
            "operates."
        ),
        has_issue=True,
        issue_description=(
            "'In perpetuity' violates client's 18-month maximum. "
            "Perpetual non-competes are also likely unenforceable in "
            "Indian courts."
        ),
        ideal_revision=(
            "Each Party agrees that during the term of the Joint Venture "
            "and for a period of eighteen (18) months following its exit "
            "or the dissolution of the Joint Venture, it shall not "
            "directly or indirectly engage in any business that competes "
            "with the Joint Venture's core operations in India."
        ),
        grading_rules={
            "concepts": {
                "eighteen_or_less": [
                    "eighteen (18) month", "18 month", "eighteen-month",
                    "eighteen month", "twelve (12) month", "12 month",
                    "twelve-month", "twelve month",
                    "six (6) month", "6 month", "six-month", "six month",
                ],
                "twenty_four_or_less": [
                    "twenty-four (24) month", "24 month",
                    "twenty-four month", "twenty (20) month",
                    "20 month", "twenty month",
                ],
                "has_month_period": [
                    "month", "months",
                ],
                "geographic_limit": [
                    "in india", "limited to india", "within india",
                    "in the territory of india", "within the territory",
                    "core operations in", "operations in india",
                ],
            },
            "forbidden_keywords": [
                "perpetuity", "perpetual", "perpetually", "forever",
            ],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["eighteen_or_less", "geographic_limit"], "score": 1.0},
                {"required": ["eighteen_or_less"], "score": 0.7},
                {"required": ["twenty_four_or_less"], "score": 0.7},
                {"required": ["has_month_period"], "score": 0.4},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_5",
        title="Expenditure Authority",
        text=(
            "Any single expenditure or commitment exceeding \u20B91 crore "
            "(INR 10,000,000) shall require prior written approval of "
            "both Parties. The Joint Venture management shall present a "
            "detailed justification and cost-benefit analysis for any "
            "such expenditure."
        ),
        has_issue=False,
    ),
    Clause(
        id="clause_6",
        title="Board Composition",
        text=(
            "The Board of Directors shall comprise five (5) members, of "
            "which four (4) shall be nominated by US TechCorp and one "
            "(1) by Bharat Digital. The Chairperson shall be nominated "
            "by US TechCorp and shall have a casting vote in the event "
            "of a tie."
        ),
        has_issue=True,
        issue_description=(
            "Client has 49% stake but only 1 of 5 seats (20% "
            "representation). Requirement is minimum 2 seats. Casting "
            "vote with US partner's chair makes it worse."
        ),
        ideal_revision=(
            "The Board of Directors shall comprise five (5) members, of "
            "which three (3) shall be nominated by US TechCorp and two "
            "(2) by Bharat Digital, reflecting proportional "
            "representation. The Chairperson shall rotate annually "
            "between the Parties. In the event of a tie, the matter "
            "shall be escalated to the deadlock resolution mechanism."
        ),
        grading_rules={
            "concepts": {
                "two_seats": [
                    "two (2)", "2 seats", "two seats",
                    "2 of 5", "two of five",
                    "nominated by bharat digital",
                    "three (3) shall be nominated by us techcorp",
                ],
                "no_casting_vote": [
                    "rotate", "rotating", "shall rotate",
                    "escalated to", "deadlock resolution",
                    "no casting vote", "without a casting vote",
                    "chairperson shall rotate", "alternating",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"required": ["two_seats", "no_casting_vote"], "score": 1.0},
                {"required": ["two_seats"], "score": 0.6},
                {"min_count": 1, "score": 0.3},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_7",
        title="Audit Rights",
        text=(
            "Each Party shall have the right to conduct an annual audit "
            "of the Joint Venture's financial records through an "
            "independent auditor of its choice, with reasonable notice "
            "of not less than fifteen (15) days. The JV entity shall "
            "provide full cooperation and access to all financial records."
        ),
        has_issue=False,
    ),
    Clause(
        id="clause_8",
        title="Exit Clause",
        text=(
            "A Party wishing to exit the Joint Venture must provide "
            "twelve (12) months written notice. The exiting Party's "
            "stake shall be valued at book value as determined by the "
            "JV's internal accounting team. Payment shall be made within "
            "one hundred eighty (180) days of the exit date."
        ),
        has_issue=True,
        issue_description=(
            "Three problems: (1) 12 months notice vs client's 60-day "
            "requirement. (2) 'Book value' instead of 'fair market "
            "value' -- undervalues the stake. (3) Internal accounting "
            "team determines value -- conflict of interest."
        ),
        ideal_revision=(
            "A Party wishing to exit the Joint Venture shall provide "
            "sixty (60) days written notice. The exiting Party's stake "
            "shall be valued at fair market value as determined by an "
            "independent valuer mutually agreed upon by the Parties. "
            "Payment shall be made within sixty (60) days of the "
            "valuation date. If the Parties cannot agree on a valuer "
            "within fifteen (15) days, each Party shall appoint one "
            "valuer and the two valuers shall appoint a third, whose "
            "determination shall be final."
        ),
        grading_rules={
            "concepts": {
                "sixty_days_notice": [
                    "sixty (60) day", "60 day", "60-day notice",
                    "sixty day", "sixty-day",
                ],
                "fair_market_value": [
                    "fair market value", "fmv", "market value",
                ],
                "independent_valuer": [
                    "independent valuer", "mutually agreed",
                    "independent valuation", "third-party valuer",
                    "independent appraiser", "independent third party",
                ],
            },
            "forbidden_keywords": [],
            "forbidden_score": 0.0,
            "tiers": [
                {"min_count": 3, "score": 1.0},
                {"min_count": 2, "score": 0.6},
                {"min_count": 1, "score": 0.3},
                {"min_count": 0, "score": 0.0},
            ],
        },
    ),
    Clause(
        id="clause_9",
        title="Governing Law",
        text=(
            "This Agreement shall be governed by the laws of India. Any "
            "disputes arising under this Agreement shall be resolved "
            "through binding arbitration under the rules of the "
            "Singapore International Arbitration Centre (SIAC), with "
            "the seat of arbitration in Singapore."
        ),
        has_issue=False,
    ),
    Clause(
        id="clause_10",
        title="Confidentiality and Data Protection",
        text=(
            "Each Party shall maintain the confidentiality of all "
            "proprietary and business information of the other Party "
            "and the Joint Venture. Neither Party shall disclose such "
            "information to any third party without prior written "
            "consent, except as required by law or regulation."
        ),
        has_issue=False,
    ),
]

JV_COUNTERPARTY_RULES: List[CounterpartyRule] = [
    CounterpartyRule(
        clause_id="clause_1",
        response_text=(
            "The JV entity must own all developed IP for clean "
            "licensing. We'll accept a revocable license on pre-existing "
            "IP."
        ),
        correct_agent_response="REJECT",
        reason="Joint ownership of developed IP is non-negotiable for the client.",
    ),
    CounterpartyRule(
        clause_id="clause_2",
        response_text="Agreed on ROFR, but with a 30-day window, not 60.",
        correct_agent_response="ACCEPT",
        reason="30-day ROFR is reasonable; client has secured the mechanism.",
    ),
    CounterpartyRule(
        clause_id="clause_3",
        response_text=(
            "Arbitration only, no buyout option. Singapore seat is fine."
        ),
        correct_agent_response="REJECT",
        reason="Buyout option is a non-negotiable client requirement.",
    ),
    CounterpartyRule(
        clause_id="clause_4",
        response_text="We can agree to 36 months, limited to India.",
        correct_agent_response="REJECT",
        reason="Push for 18 months -- 36 months exceeds client's hard limit.",
    ),
    CounterpartyRule(
        clause_id="clause_6",
        response_text=(
            "Two (2) seats for Bharat Digital, yes -- but the chair "
            "stays with us and retains a casting vote."
        ),
        correct_agent_response="REJECT",
        reason="Push for rotating chair or no casting vote.",
    ),
    CounterpartyRule(
        clause_id="clause_8",
        response_text=(
            "90 days notice, fair market value, independent valuer. "
            "Payment within 120 days."
        ),
        correct_agent_response="ACCEPT",
        reason="Close enough to client tolerance -- core protections are in place.",
    ),
]

JV_SCENARIO = Scenario(
    task_id="jv",
    contract_type="Joint Venture Agreement",
    client_name="Bharat Digital Infrastructure Ltd.",
    client_role="Joint Venture Partner (49% stake)",
    client_requirements=[
        "All IP developed during the JV belongs jointly; pre-existing IP remains with original owner.",
        "Client must have right of first refusal if partner wants to sell stake.",
        "Deadlock resolution mechanism must include buyout option, not just arbitration.",
        "Non-compete must be limited to 18 months post-exit, not perpetual.",
        "Client's approval required for any expenditure exceeding \u20B91 crore.",
        "Board representation proportional to stake (49% = 2 of 5 seats minimum).",
        "Annual audit rights with independent auditor of client's choice.",
        "Exit clause: client can exit with fair market value within 60 days.",
        "Governing law: India, arbitration in Singapore for international disputes.",
    ],
    max_steps=35,
    clauses=JV_CLAUSES,
    counterparty_rules=JV_COUNTERPARTY_RULES,
    rubric=ScoringRubric(
        flag_issue_reward=0.05,
        false_flag_penalty=-0.02,
        revision_weight=0.06,
        accept_clean_reward=0.01,
        wrong_flag_clean_penalty=-0.02,
        completeness_bonus=0.06,
        negotiation_correct_reward=0.03,
        negotiation_wrong_penalty=-0.03,
        cross_clause_bonus=0.06,
    ),
)


# ===========================================================================
# Scenario registry
# ===========================================================================

SCENARIOS: Dict[str, Scenario] = {
    "nda": NDA_SCENARIO,
    "saas": SAAS_SCENARIO,
    "jv": JV_SCENARIO,
}


def get_scenario(task_id: str) -> Scenario:
    """Return the scenario matching ``task_id``.

    Raises:
        KeyError: if ``task_id`` is not one of "nda", "saas", "jv".
    """
    if task_id not in SCENARIOS:
        raise KeyError(
            f"Unknown task_id: {task_id!r}. "
            f"Expected one of: {sorted(SCENARIOS.keys())}"
        )
    return SCENARIOS[task_id]


def get_counterparty_rule(
    scenario: Scenario, clause_id: str
) -> Optional[CounterpartyRule]:
    """Return the counterparty rule for ``clause_id`` in ``scenario``, if any."""
    for rule in scenario.counterparty_rules:
        if rule.clause_id == clause_id:
            return rule
    return None
