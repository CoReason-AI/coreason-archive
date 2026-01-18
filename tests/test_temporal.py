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
from datetime import datetime, timedelta, timezone

from coreason_archive.models import MemoryScope
from coreason_archive.temporal import DECAY_RATES, TemporalRanker


def test_decay_calculation_basic() -> None:
    """Test basic decay calculation for a known time delta."""
    scope = MemoryScope.USER
    decay_rate = DECAY_RATES[scope]

    # 1 day ago
    delta_seconds = 86400
    now = datetime.now(timezone.utc)
    created_at = now - timedelta(seconds=delta_seconds)

    # Re-calculate delta based on actual 'now' usage inside function is not possible without mocking.
    # The function calls datetime.now(timezone.utc) internally.
    # So there is a slight time drift between our `now` and the function's `now`.
    # We relax the tolerance to account for execution time.

    expected_decay = math.exp(-decay_rate * delta_seconds)
    actual_decay = TemporalRanker.calculate_decay_factor(scope, created_at)

    assert math.isclose(actual_decay, expected_decay, rel_tol=1e-5)


def test_scope_differences() -> None:
    """Verify that USER scope decays faster than CLIENT scope."""
    delta = timedelta(days=30)
    created_at = datetime.now(timezone.utc) - delta

    user_decay = TemporalRanker.calculate_decay_factor(MemoryScope.USER, created_at)
    client_decay = TemporalRanker.calculate_decay_factor(MemoryScope.CLIENT, created_at)

    # USER memory should be retained LESS (lower factor) than CLIENT memory
    assert user_decay < client_decay


def test_zero_time_delta() -> None:
    """Test decay at the exact moment of creation (should be 1.0)."""
    now = datetime.now(timezone.utc)
    decay = TemporalRanker.calculate_decay_factor(MemoryScope.USER, now)
    assert math.isclose(decay, 1.0, rel_tol=1e-9)


def test_future_date() -> None:
    """Test that future dates are clamped to 'now' (factor 1.0)."""
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    decay = TemporalRanker.calculate_decay_factor(MemoryScope.USER, future)
    assert math.isclose(decay, 1.0, rel_tol=1e-9)


def test_adjust_score() -> None:
    """Test the full score adjustment."""
    scope = MemoryScope.PROJECT
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    original_score = 0.9

    decay_factor = TemporalRanker.calculate_decay_factor(scope, created_at)
    expected_score = original_score * decay_factor

    actual_score = TemporalRanker.adjust_score(original_score, scope, created_at)
    assert math.isclose(actual_score, expected_score, rel_tol=1e-9)


def test_naive_datetime_handling() -> None:
    """Test that naive datetimes are treated as UTC."""
    # The ranker assumes UTC for "now", so if we pass a naive time that is local "now",
    # and local != UTC, there might be a delta.
    # To test the MECHANISM safely, we should construct a naive time based on UTC now to control the test.

    utc_now = datetime.now(timezone.utc)
    # create a naive object with the same values
    naive_equivalent = utc_now.replace(tzinfo=None)

    # This should be treated as roughly "now"
    decay = TemporalRanker.calculate_decay_factor(MemoryScope.USER, naive_equivalent)

    # Allow small tolerance for execution time
    assert decay > 0.9999


def test_very_old_memory() -> None:
    """Test stability with very old memories (e.g., 10 years)."""
    scope = MemoryScope.USER  # Fast decay
    # 10 years ago
    created_at = datetime.now(timezone.utc) - timedelta(days=365 * 10)

    decay = TemporalRanker.calculate_decay_factor(scope, created_at)

    # Should be effectively zero, but definitely not negative and not crashing
    assert 0.0 <= decay < 0.01

    # Check CLIENT scope (slow decay)
    scope_client = MemoryScope.CLIENT
    decay_client = TemporalRanker.calculate_decay_factor(scope_client, created_at)
    # Should still be small but measurable?
    # lambda ~ 2e-8 * 3.15e8 (10 yrs) ~ 6.3
    # exp(-6.3) ~ 0.0018
    assert decay_client > decay


def test_negative_score_decay() -> None:
    """Test that negative scores (dissimilarity) decay towards zero (neutrality)."""
    scope = MemoryScope.USER
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    original_score = -0.5

    adjusted_score = TemporalRanker.adjust_score(original_score, scope, created_at)

    # Should be closer to 0 than -0.5, but still negative
    assert -0.5 < adjusted_score <= 0.0

    # Verify math: -0.5 * factor (where 0 < factor < 1) -> negative number smaller in magnitude
    factor = TemporalRanker.calculate_decay_factor(scope, created_at)
    assert math.isclose(adjusted_score, original_score * factor, rel_tol=1e-9)


def test_ranking_flip_complex_scenario() -> None:
    """
    Test a scenario where a high-scoring but fast-decaying memory (USER)
    is eventually outranked by a lower-scoring but slow-decaying memory (CLIENT).

    Item A (USER): Initial Score 1.0
    Item B (CLIENT): Initial Score 0.8

    Initially: A > B
    After Time T: B > A
    """
    now = datetime.now(timezone.utc)

    # Create items "virtually" at same time 'now' (simulated context)
    # Actually, let's simulate that 'now' is T hours in the future.
    # So we set created_at to T hours in the past.

    # Based on calculation, flip happens around ~8 hours
    short_time = timedelta(hours=1)
    long_time = timedelta(hours=10)

    # Time 1: 1 hour elapsed
    created_at_1 = now - short_time

    score_a_1 = TemporalRanker.adjust_score(1.0, MemoryScope.USER, created_at_1)
    score_b_1 = TemporalRanker.adjust_score(0.8, MemoryScope.CLIENT, created_at_1)

    # A should still be winning
    assert score_a_1 > score_b_1, f"At 1 hour, USER ({score_a_1}) should beat CLIENT ({score_b_1})"

    # Time 2: 10 hours elapsed
    created_at_2 = now - long_time

    score_a_2 = TemporalRanker.adjust_score(1.0, MemoryScope.USER, created_at_2)
    score_b_2 = TemporalRanker.adjust_score(0.8, MemoryScope.CLIENT, created_at_2)

    # B should now be winning
    assert score_b_2 > score_a_2, f"At 10 hours, CLIENT ({score_b_2}) should beat USER ({score_a_2})"
