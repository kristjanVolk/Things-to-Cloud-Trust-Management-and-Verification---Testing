import math
import pytest

from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig


def test_bad_mouthing_reduces_trust():
    model = AdaptiveTrust(config=TrustConfig())

    for _ in range(30):
        model.update(1.0)

    baseline = model.trust_score()

    for _ in range(10):
        model.update(0.0)

    attacked = model.trust_score()

    assert attacked < baseline


def test_bad_mouthing_changes_mean_behavior():
    model = AdaptiveTrust(config=TrustConfig(window_size=5))

    for _ in range(20):
        model.update(1.0)

    stable = model.trust_score()

    for _ in range(10):
        model.update(0.0)

    attacked = model.trust_score()

    assert attacked < stable


def test_lambda_adapts_under_bad_mouthing():
    model = AdaptiveTrust(
        config=TrustConfig(base_lambda=0.2, window_size=5)
    )

    for _ in range(20):
        model.update(1.0)

    baseline_lambda = model.compute_lambda()

    for _ in range(10):
        model.update(0.0)

    attacked_lambda = model.compute_lambda()

    assert attacked_lambda >= baseline_lambda
    assert 0.0 <= attacked_lambda <= 1.0


def test_trust_degrades_but_stays_bounded():
    model = AdaptiveTrust(config=TrustConfig())

    for _ in range(50):
        model.update(1.0)

    for _ in range(50):
        model.update(0.0)

    score = model.trust_score()

    assert 0.0 <= score <= 1.0


def test_adaptive_recovery_after_bad_mouthing():
    model = AdaptiveTrust(config=TrustConfig())

    # build trust
    for _ in range(30):
        model.update(1.0)

    peak = model.trust_score()

    # attack phase
    for _ in range(15):
        model.update(0.0)

    attacked = model.trust_score()

    # recovery phase (honest again)
    for _ in range(10):
        model.update(1.0)

    recovered = model.trust_score()

    assert attacked < peak
    assert recovered > attacked
    assert recovered <= peak  