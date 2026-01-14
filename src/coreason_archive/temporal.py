# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_archive

import math
from datetime import datetime, timezone

from coreason_archive.models import MemoryScope

# Decay constants (Lambda) per scope
# Higher value = Faster decay
# Time unit for delta_t is SECONDS.
# Example: If lambda = 1e-6, decay factor over 1 day (86400s) is exp(-1e-6 * 86400) ~= 0.917
#
# Assumptions:
# USER: Fast decay (scratchpad/context). Half-life ~1 day -> lambda ~ 8e-6
# PROJECT: Medium decay. Half-life ~1 month -> lambda ~ 2.6e-7
# DEPT: Slow decay. Half-life ~6 months -> lambda ~ 4e-8
# CLIENT: Very slow decay. Half-life ~1 year -> lambda ~ 2e-8

DECAY_RATES: dict[MemoryScope, float] = {
    MemoryScope.USER: 8.0e-6,
    MemoryScope.PROJECT: 2.6e-7,
    MemoryScope.DEPARTMENT: 4.0e-8,
    MemoryScope.CLIENT: 2.0e-8,
}


class TemporalRanker:
    """
    Adjusts similarity scores based on recency (Recency Bias) using the formula:
    S_final = S_vector * e^(-lambda * delta_t)
    """

    @staticmethod
    def calculate_decay_factor(scope: MemoryScope, created_at: datetime) -> float:
        """
        Calculates the decay factor e^(-lambda * delta_t).

        Args:
            scope: The scope of the memory (determines lambda).
            created_at: The creation timestamp of the memory.

        Returns:
            A float between 0.0 and 1.0 representing the decay multiplier.
        """
        # Ensure created_at is timezone-aware (assume UTC if naive, though models enforce awareness)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        delta_t_seconds = (now - created_at).total_seconds()

        # Handle edge case: Future dates or exact current time
        if delta_t_seconds < 0:
            # Treat future dates as "now" (no decay) or log warning?
            # For robustness, we clamp to 0.
            delta_t_seconds = 0.0

        decay_rate = DECAY_RATES.get(scope, 0.0)

        # Calculate exponential decay
        return math.exp(-decay_rate * delta_t_seconds)

    @staticmethod
    def adjust_score(vector_score: float, scope: MemoryScope, created_at: datetime) -> float:
        """
        Adjusts the vector similarity score based on time decay.

        Args:
            vector_score: The original similarity score (usually 0.0 to 1.0).
            scope: The scope of the memory.
            created_at: The creation timestamp.

        Returns:
            The adjusted score.
        """
        decay_factor = TemporalRanker.calculate_decay_factor(scope, created_at)
        return vector_score * decay_factor
