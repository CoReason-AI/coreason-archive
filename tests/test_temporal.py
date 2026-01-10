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
    created_at = datetime.now(timezone.utc) - timedelta(seconds=delta_seconds)

    expected_decay = math.exp(-decay_rate * delta_seconds)
    actual_decay = TemporalRanker.calculate_decay_factor(scope, created_at)

    assert math.isclose(actual_decay, expected_decay, rel_tol=1e-9)


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
