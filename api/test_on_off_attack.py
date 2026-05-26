import statistics
import pytest
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig



#! Verify that lambda stays within bounds [0,1] regardless of volatility
def test_compute_lambda_bounds():
    config = TrustConfig(base_lambda=0.2, window_size=1)  # window_size=1 == zero volatility
    model = AdaptiveTrust(config=config)
    # No volatility -> lambda should be base_lambda
    lam = model.compute_lambda()
    assert 0.0 <= lam <= 1.0
    assert lam == pytest.approx(config.base_lambda)
    # Force high volatility by feeding a large range of observations
    model.recent_observations = [0.0, 1.0]
    expected = min(1.0, config.base_lambda + statistics.stdev(model.recent_observations) / 2)
    lam = model.compute_lambda()
    assert lam == pytest.approx(expected)
    model.recent_observations = [0.0, 1.0] * 100
    lam = model.compute_lambda()
    assert lam <= 1.0


# Trust score should always stay between 0 and 1

def test_trust_score_bounds():
    model = AdaptiveTrust(config=TrustConfig())
    for v in [0.0, 0.5, 1.0, -0.1, 1.1]:
        # Update with out-of-bounds values should still produce a trust score in [0,1]
        model.update(v)
        score = model.trust_score()
        assert 0.0 <= score <= 1.0


# Adjusted trust score must also be bounded

def test_adjusted_trust_score_bounds():
    model = AdaptiveTrust(config=TrustConfig())
    # Simulate a scenario with high variance to trigger penalty
    model.recent_observations = [0.0, 1.0]
    model.update(0.0)
    adj = model.adjusted_trust_score()
    assert 0.0 <= adj <= 1.0


# Test round‑trip serialization

def test_roundtrip_dict_and_json():
    model = AdaptiveTrust(config=TrustConfig())
    for v in [0.3, 0.7, 0.9]:
        model.update(v)
    data = model.to_dict()
    new_model = AdaptiveTrust(config=TrustConfig()).from_dict(data)
    assert new_model.trust_score() == pytest.approx(model.trust_score())
    assert new_model.compute_lambda() == pytest.approx(model.compute_lambda())
    assert list(new_model.recent_observations) == list(model.recent_observations)


# Compute volatility with constant observations returns 0

def test_constant_observations_volatility():
    model = AdaptiveTrust(config=TrustConfig(window_size=5))
    const = [0.8] * 5
    for v in const:
        model.update(v)
    assert model.compute_volatility() == 0.0

def test_on_off_attack_reduces_trust():
    model = AdaptiveTrust(config=TrustConfig())

    scores = []

    # attacker first behaves well
    for _ in range(20):
        model.update(1.0)
        scores.append(model.trust_score())

    high_trust = model.trust_score()

    # then starts malicious behavior
    for _ in range(10):
        model.update(0.0)
        scores.append(model.trust_score())

    low_trust = model.trust_score()

    # trust should decrease significantly
    assert low_trust < high_trust
    assert low_trust < 0.05
    

# Verify that the volatility function uses only the last window_size observations
def test_volatility_window_limit():
    config = TrustConfig(window_size=3)
    model = AdaptiveTrust(config=config)
    observations = [0.1, 0.4, 0.9, 0.3]
    for v in observations:
        model.update(v)
    # Only last 3 observations should be used
    expected_observations = [0.4, 0.9, 0.3]
    expected_vol = statistics.stdev(expected_observations)

    assert model.compute_volatility() == pytest.approx(expected_vol)


def test_on_off_attack():
    model = AdaptiveTrust(config=TrustConfig())
    # attacker gains trust
    for _ in range(20):
        model.update(1.0)
    trust_before_attack = model.trust_score()
    # first attack
    for _ in range(5):
        model.update(0.0)
    trust_after_attack = model.trust_score()
    # recover behavior
    for _ in range(5):
        model.update(1.0)
    trust_after_recovery = model.trust_score()
    # second attack
    for _ in range(5):
        model.update(0.0)
    trust_after_second_attack = model.trust_score()
    assert trust_after_attack < trust_before_attack
    # recovery should not be immediate
    assert trust_after_recovery < trust_before_attack
    # repeated attacks should keep trust low
    assert trust_after_second_attack < trust_after_recovery


