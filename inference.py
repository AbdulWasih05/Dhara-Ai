# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline agent for the Dhara AI contract-negotiation environment.

Runs a single-shot LLM agent against all three tasks (nda, saas, jv)
sequentially and emits the structured [START]/[STEP]/[END] stdout
required by the hackathon grading harness.

Environment variables (read at startup):
    HF_TOKEN       -- HuggingFace router API key (required)
    API_BASE_URL   -- OpenAI-compatible base URL
                      default: "https://router.huggingface.co/v1"
    MODEL_NAME     -- Chat model id
                      default: "Qwen/Qwen2.5-72B-Instruct"
    IMAGE_NAME     -- Docker image for the environment server (optional).
                      If set, the client boots the container.
    SPACE_URL      -- Base URL of a running environment server
                      (HF Space, or local). Used when IMAGE_NAME is unset.
                      default: "http://127.0.0.1:8000"

Usage (local dev):
    # terminal 1: start the environment server
    uv run --project dhara_ai server
    # terminal 2:
    export HF_TOKEN=hf_...
    python dhara_ai/inference.py

Stdout contract (must be exact, matching CLAUDE.md):
    [START] task=<name> env=dhara_ai model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
import time
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# Support both "run from parent dir" and "run from dhara_ai/" invocations.
try:
    from dhara_ai.client import DharaAiEnv
    from dhara_ai.models import DharaAiAction, DharaAiObservation
except ModuleNotFoundError:  # pragma: no cover -- invoked as `python inference.py`
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import DharaAiEnv  # type: ignore
    from models import DharaAiAction, DharaAiObservation  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# CRITICAL FIX: Use platform-injected environment variables without defaults
# The hackathon validator requires these specific variables to route through
# the LiteLLM proxy. Do NOT use fallback values.
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")  # Changed from HF_TOKEN to API_KEY
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "").strip()
SPACE_URL = os.environ.get("SPACE_URL", "http://127.0.0.1:7860").strip()

# Startup config logging (stderr — does not affect stdout parsing)
print(f"[CONFIG] API_BASE_URL={API_BASE_URL}", file=sys.stderr)
print(f"[CONFIG] MODEL_NAME={MODEL_NAME}", file=sys.stderr)
print(f"[CONFIG] HF_TOKEN_present={HF_TOKEN is not None}", file=sys.stderr)
print(f"[CONFIG] IMAGE_NAME={IMAGE_NAME!r}", file=sys.stderr)
print(f"[CONFIG] SPACE_URL={SPACE_URL}", file=sys.stderr)

TASKS: List[str] = ["nda", "saas", "jv"]

# Upper bound on iterations as a safety net. The environment enforces
# its own max_steps (15/25/35) and auto-submits, so the loop should
# terminate naturally well before this fires.
MAX_ITERATIONS_PER_TASK = 60

# LLM call parameters. Low temperature -> more deterministic behaviour;
# short max_tokens is fine because the agent only needs to emit one
# compact JSON object per turn.
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = textwrap.dedent(
    """\
    You are a legal contract review agent. You are reviewing a {contract_type}
    on behalf of your client ({client_role}).

    CLIENT REQUIREMENTS:
    {requirements_block}

    AVAILABLE ACTIONS -- respond with EXACTLY ONE action per turn, as a
    single JSON object. No explanation, no markdown, no extra text.

      1. flag_issue
         {{"action_type": "flag_issue", "clause_id": "<id>", "issue": "<short description>"}}

      2. suggest_revision
         {{"action_type": "suggest_revision", "clause_id": "<id>", "revised_text": "<new full clause text>"}}

      3. accept_clause
         {{"action_type": "accept_clause", "clause_id": "<id>"}}

      4. request_clarification
         {{"action_type": "request_clarification", "clause_id": "<id>", "question": "<question>"}}

      5. submit_review
         {{"action_type": "submit_review"}}

    PROCEDURE:
      - Walk through the clauses in order.
      - Flag any clause that contradicts or fails to meet a client requirement.
      - After flagging, call suggest_revision and draft replacement text that
        satisfies the requirement and mentions the specific terms the client
        cares about (e.g. exact durations, "India", "Bengaluru", dollar/rupee
        caps, "fair market value", "independent valuer", etc.).
      - Accept clauses that already satisfy the requirements as-is.
      - If the counterparty pushes back on a revision, look at the client's
        hard limits. ACCEPT their counter-offer (call accept_clause) only if
        it still meets the client's requirements; otherwise REJECT by calling
        suggest_revision again with your preferred wording.
      - When every clause is addressed (accepted or revised), call submit_review.

    Respond with ONLY a JSON action object. Nothing else.
    """
)


def _build_system_prompt(obs: DharaAiObservation) -> str:
    reqs = "\n".join(
        f"  {i + 1}. {req}" for i, req in enumerate(obs.client_requirements)
    )
    return _SYSTEM_TEMPLATE.format(
        contract_type=obs.contract_type or "contract",
        client_role=obs.client_role or "the client",
        requirements_block=reqs or "  (none specified)",
    )


def _build_user_message(obs: DharaAiObservation) -> str:
    lines: List[str] = []
    lines.append(
        f"STATUS: {obs.review_progress} | {obs.steps_remaining} steps remaining"
    )
    lines.append("")
    lines.append("CLAUSES:")
    for c in obs.clauses:
        lines.append(f"[{c.id}] {c.title} -- status: {c.status}")
        lines.append(f"    {c.text}")
        if c.negotiation_history:
            for h in c.negotiation_history:
                lines.append(f"    > {h}")
        lines.append("")
    if obs.last_action_result:
        lines.append(f"Last action result: {obs.last_action_result}")
    if obs.last_action_error:
        lines.append(f"Last action error: {obs.last_action_error}")
    lines.append("")
    lines.append("Respond with ONE action as a JSON object.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def _strip_markdown_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # Drop the opening fence line (e.g. ```json or ```)
        newline = t.find("\n")
        if newline != -1:
            t = t[newline + 1 :]
        # Drop the closing fence, if present
        if "```" in t:
            t = t[: t.rfind("```")]
    return t.strip()


def _parse_action(text: str) -> Optional[DharaAiAction]:
    """Extract a DharaAiAction from raw LLM output.

    Tolerant of markdown fences, leading/trailing commentary, and extra
    whitespace. Returns None if no valid action can be parsed.
    """
    if not text:
        return None
    cleaned = _strip_markdown_fence(text)
    # Find the outermost JSON object in the cleaned text.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return DharaAiAction.model_validate(data)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Log formatting
# ---------------------------------------------------------------------------


def _bool_str(b: bool) -> str:
    return "true" if b else "false"


def _action_str(action: Optional[DharaAiAction]) -> str:
    if action is None:
        return "PARSE_ERROR"
    atype = action.action_type
    if atype == "submit_review":
        return "submit_review"
    return f"{atype}({action.clause_id or 'none'})"


def _error_str(err: Optional[str]) -> str:
    if err is None:
        return "null"
    # Keep the error inline: strip newlines so log lines stay single-line.
    return " ".join(err.split())


def _log(line: str) -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Environment client factory
# ---------------------------------------------------------------------------


def _make_env() -> DharaAiEnv:
    """Instantiate the environment client.

    If IMAGE_NAME is set, spin up a fresh Docker container and connect
    to it. Otherwise connect to an already-running server at SPACE_URL
    (HF Space, or a local dev server).
    """
    print(f"[ENV] connecting via image_name={IMAGE_NAME!r} space_url={SPACE_URL!r}", file=sys.stderr, flush=True)
    if IMAGE_NAME:
        env = DharaAiEnv.from_docker_image(IMAGE_NAME)
    else:
        env = DharaAiEnv(base_url=SPACE_URL)
    print(f"[ENV] client created", file=sys.stderr, flush=True)
    return env


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_last_call_time = 0.0
_MIN_CALL_INTERVAL = 3.0  # seconds between calls (30 RPM for Groq = 2s, +1s buffer)


def _chat(
    llm: OpenAI, model: str, system_prompt: str, user_message: str
) -> str:
    """Synchronous chat completion with rate-limit retry. Returns assistant text."""
    global _last_call_time

    for attempt in range(5):
        # Enforce minimum interval between calls
        elapsed = time.time() - _last_call_time
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)

        _last_call_time = time.time()
        print(f"[LLM_CALL] attempt={attempt+1} model={model} base_url={llm.base_url}", file=sys.stderr, flush=True)
        try:
            response = llm.chat.completions.create(
                model=model,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            print(f"[LLM_CALL_OK] attempt={attempt+1}", file=sys.stderr, flush=True)
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(f"[LLM_CALL_FAIL] attempt={attempt+1} type={type(exc).__name__} msg={str(exc)[:200]}", file=sys.stderr, flush=True)
            err_str = str(exc)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                match = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
                wait = float(match.group(1)) + 2 if match else 15.0
                print(f"[INFO] Rate limited, waiting {wait:.0f}s (attempt {attempt+1}/5)", flush=True)
                time.sleep(wait)
                continue
            if "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 15.0
                print(f"[INFO] Service unavailable, retrying in {wait:.0f}s (attempt {attempt+1}/5)", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Rate limit exceeded after 5 retries")


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


async def _run_task(task_name: str, llm: OpenAI) -> None:
    _log(f"[START] task={task_name} env=dhara_ai model={MODEL_NAME}")

    env = _make_env()
    rewards: List[float] = []
    step_n = 0
    final_done = False

    try:
        async with env:
            print(f"[ENV_RESET] task={task_name}", file=sys.stderr, flush=True)
            result = await env.reset(task_name=task_name)
            observation = result.observation
            print(f"[ENV_RESET_OK] task={task_name} clauses={len(observation.clauses)}", file=sys.stderr, flush=True)
            system_prompt = _build_system_prompt(observation)

            while step_n < MAX_ITERATIONS_PER_TASK:
                if result.done:
                    final_done = True
                    break

                user_message = _build_user_message(observation)

                # Call the LLM; if it errors (network/timeout), emit a
                # submit_review fallback so the episode doesn't hang.
                try:
                    raw = _chat(
                        llm, MODEL_NAME, system_prompt, user_message
                    )
                except Exception as exc:
                    print(f"[EXCEPT] location=run_task_llm_call task={task_name} step={step_n+1} type={type(exc).__name__} msg={str(exc)[:200]}", file=sys.stderr, flush=True)
                    step_n += 1
                    reason = f"llm_error: {exc}"
                    rewards.append(0.0)
                    _log(
                        f"[STEP]  step={step_n} action=submit_review "
                        f"reward=0.00 done=true error={_error_str(reason)}"
                    )
                    try:
                        result = await env.step(
                            DharaAiAction(action_type="submit_review")
                        )
                    except Exception as e2:
                        print(f"[EXCEPT] location=fallback_submit task={task_name} type={type(e2).__name__} msg={str(e2)[:200]}", file=sys.stderr, flush=True)
                    final_done = True
                    break

                parsed = _parse_action(raw)
                if parsed is None:
                    print(f"[DEBUG] Failed to parse LLM response: {repr(raw[:500])}", flush=True)
                    parsed = DharaAiAction(action_type="submit_review")

                step_n += 1
                print(f"[ENV_STEP] task={task_name} step={step_n} action={_action_str(parsed)}", file=sys.stderr, flush=True)
                result = await env.step(parsed)
                reward = float(result.reward or 0.0)
                print(f"[ENV_STEP_OK] task={task_name} step={step_n} reward={reward:.2f} done={result.done}", file=sys.stderr, flush=True)
                rewards.append(reward)
                observation = result.observation

                _log(
                    f"[STEP]  step={step_n} "
                    f"action={_action_str(parsed)} "
                    f"reward={reward:.2f} "
                    f"done={_bool_str(bool(result.done))} "
                    f"error={_error_str(observation.last_action_error)}"
                )

                if result.done:
                    final_done = True
                    break
    except Exception as exc:
        print(f"[EXCEPT] location=run_task_outer task={task_name} type={type(exc).__name__} msg={str(exc)[:200]}", file=sys.stderr, flush=True)
        # Defensive: never crash without emitting [END].
        _log(
            f"[STEP]  step={step_n + 1} action=ERROR reward=0.00 done=true "
            f"error={_error_str(f'runtime_error: {exc}')}"
        )

    score = max(0.0, min(1.0, sum(rewards)))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    _log(
        f"[END]   success={_bool_str(final_done)} steps={step_n} "
        f"score={score:.2f} rewards={rewards_str}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _amain() -> None:
    # CRITICAL FIX: Check for API_KEY instead of HF_TOKEN (platform-injected variable)
    if not API_KEY or not API_BASE_URL:
        # Emit a fake [START]/[END] for each task so the harness still
        # sees the required markers, then exit.
        for task in TASKS:
            _log(f"[START] task={task} env=dhara_ai model={MODEL_NAME}")
            _log(
                "[STEP]  step=1 action=NONE reward=0.00 done=true "
                "error=API_KEY or API_BASE_URL environment variables are not set"
            )
            _log(
                "[END]   success=false steps=0 score=0.00 rewards="
            )
        return

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name in TASKS:
        await _run_task(task_name, llm)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
