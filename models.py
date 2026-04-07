# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Top-level model re-exports for the Dhara AI environment.

This module provides package-level aliases for the contract-negotiation
types defined in ``dhara_ai.server.models`` so that external callers
can do:

    from dhara_ai import DharaAiAction, DharaAiObservation

The underlying Pydantic models are shared across client and server.
"""

try:
    from .server.models import (
        ActionType,
        ClauseInfo,
        ClauseState,
        ClauseStatus,
        ContractAction,
        ContractObservation,
        ContractState,
    )
except ImportError:  # pragma: no cover -- non-package execution fallback
    from server.models import (  # type: ignore
        ActionType,
        ClauseInfo,
        ClauseState,
        ClauseStatus,
        ContractAction,
        ContractObservation,
        ContractState,
    )

# Package-level aliases (match the dhara_ai package name)
DharaAiAction = ContractAction
DharaAiObservation = ContractObservation
DharaAiState = ContractState

__all__ = [
    # Primary contract names
    "ContractAction",
    "ContractObservation",
    "ContractState",
    "ClauseInfo",
    "ClauseState",
    "ClauseStatus",
    "ActionType",
    # Package-name aliases
    "DharaAiAction",
    "DharaAiObservation",
    "DharaAiState",
]
