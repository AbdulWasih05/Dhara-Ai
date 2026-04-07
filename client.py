# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dhara AI Environment Client.

Thin EnvClient subclass that wires up Pydantic serialization for the
ContractAction / ContractObservation / ContractState types. Uses a
persistent WebSocket session so state survives across reset() / step()
calls.
"""

import os
from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from websockets import connect as ws_connect

try:
    from .models import DharaAiAction, DharaAiObservation, DharaAiState
except ImportError:  # pragma: no cover -- non-package execution fallback
    from models import DharaAiAction, DharaAiObservation, DharaAiState  # type: ignore


class DharaAiEnv(EnvClient[DharaAiAction, DharaAiObservation, DharaAiState]):
    """
    Client for the Dhara AI contract-negotiation environment.

    Each client instance holds its own WebSocket session, which keeps a
    dedicated server-side environment instance alive across reset() and
    step() calls.

    Example (local dev server):
        >>> env = DharaAiEnv(base_url="http://localhost:8000")
        >>> async with env:
        ...     result = await env.reset(task_name="nda")
        ...     print(result.observation.contract_type)
        ...     result = await env.step(DharaAiAction(
        ...         action_type="flag_issue",
        ...         clause_id="clause_1",
        ...         issue="definition too narrow",
        ...     ))

    Example (Docker container):
        >>> env = DharaAiEnv.from_docker_image("dhara_ai:latest")
        >>> async with env:
        ...     result = await env.reset(task_name="saas")
    """

    async def connect(self) -> "DharaAiEnv":
        """Connect with extended ping timeouts to survive long LLM waits."""
        if self._ws is not None:
            return self

        ws_url_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_url_lower or "127.0.0.1" in ws_url_lower

        old_no_proxy = os.environ.get("NO_PROXY")
        if is_localhost:
            current_no_proxy = old_no_proxy or ""
            if "localhost" not in current_no_proxy.lower():
                os.environ["NO_PROXY"] = (
                    f"{current_no_proxy},localhost,127.0.0.1"
                    if current_no_proxy
                    else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
                ping_interval=120,
                ping_timeout=300,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {e}") from e
        finally:
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    def _step_payload(self, action: DharaAiAction) -> Dict[str, Any]:
        """Serialize a ContractAction to the JSON payload shape expected
        by the server's /step (WebSocket) handler.

        We drop default ``metadata={}`` to keep messages tight, and we
        explicitly emit ``None`` for optional fields the server reads.
        """
        payload: Dict[str, Any] = {"action_type": action.action_type}
        # Include optional fields only if set (keeps wire payload tight)
        if action.clause_id is not None:
            payload["clause_id"] = action.clause_id
        if action.issue is not None:
            payload["issue"] = action.issue
        if action.revised_text is not None:
            payload["revised_text"] = action.revised_text
        if action.question is not None:
            payload["question"] = action.question
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[DharaAiObservation]:
        """Parse a server observation response into a StepResult.

        The server returns ``{"observation": {...}, "reward": float,
        "done": bool}`` inside the WebSocket message's ``data`` field.
        """
        obs_data = payload.get("observation", {}) or {}
        # Use pydantic's validator to keep type safety
        observation = DharaAiObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DharaAiState:
        """Parse a server /state response into a DharaAiState."""
        return DharaAiState.model_validate(payload)
