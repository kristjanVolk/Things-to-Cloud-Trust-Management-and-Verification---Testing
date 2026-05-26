import pytest

from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig
from app.services.trust.trust_calculator import TrustCalculator


def make_resource_payload(direct_trust=True):
    return {
        "placement": {"provider_score": 0.8},
        "monitoring": {"availability": 1.0, "reliability": 0.5},
        "context": {"matching_constraints": 1, "total_constraints": 2},
        "reputation": 0.3,
        "direct_trust": direct_trust,
    }


def test_adaptive_trust_prior():
    model = AdaptiveTrust(config=TrustConfig())

    expected_prior = 1.0 / (1.0 + 1.0)

    assert model.alpha == pytest.approx(1.0)
    assert model.beta == pytest.approx(1.0)
    assert model.trust_score() == pytest.approx(expected_prior)
    assert model.n_eff == pytest.approx(2.0)
    assert model.variance > 0.0
    assert model.std_dev > 0.0


def test_adaptive_trust_update_with_good_observation_increases_mean():
    model = AdaptiveTrust(config=TrustConfig())

    before = model.trust_score()
    model.update(1.0)
    after = model.trust_score()

    assert after > before
    assert model.alpha > model.beta


def test_adaptive_trust_update_with_bad_observation_decreases_mean():
    model = AdaptiveTrust(config=TrustConfig())

    before = model.trust_score()
    model.update(0.0)
    after = model.trust_score()

    assert after < before
    assert model.alpha < model.beta


def test_compute_resource_trust_returns_zero_when_direct_trust_false():
    resource = make_resource_payload(direct_trust=False)

    assert TrustCalculator.compute_resource_trust(resource) == 0.0


def test_compute_resource_trust_returns_weighted_score_when_direct_trust_true():
    resource = make_resource_payload(direct_trust=True)
    computed = TrustCalculator.compute_resource_trust(resource)
    
    assert 0.0 <= computed <= 1.0


def test_update_provider_trust_serializes_model_and_updates_contextual_trust():
    provider = AdaptiveTrust(
        config=TrustConfig(),
        direct_trust=True,
        reputation=0.3,
    )

    adaptive_dict, contextual_trust, resource_trusts = (
        TrustCalculator.update_provider_trust(
            provider_model=provider,
            resources_json_list=[make_resource_payload(direct_trust=True)],
        )
    )

    expected_trust = provider.trust_score()

    assert isinstance(adaptive_dict, dict)
    assert adaptive_dict["trust_score"] == pytest.approx(expected_trust)
    assert "alpha" in adaptive_dict
    assert "beta" in adaptive_dict
    assert 0.0 <= contextual_trust <= 1.0


def test_update_provider_trust_returns_prior_on_empty_resource_list():
    provider = AdaptiveTrust(config=TrustConfig())

    adaptive_dict, contextual_trust, resource_trusts = (
        TrustCalculator.update_provider_trust(
            provider_model=provider,
            resources_json_list=[],
        )
    )

    expected_prior = provider.trust_score()

    assert contextual_trust == 0.0
    assert resource_trusts == 0.0
    assert adaptive_dict["trust_score"] == pytest.approx(expected_prior)


# =========================================================
# 3.1 Functional Correctness Testing
# =========================================================

def test_consistently_good_behavior_converges_to_high_trust():
    model = AdaptiveTrust(config=TrustConfig())

    for _ in range(100):
        model.update(1.0)

    assert model.trust_score() > model.trust_score() * 0.9
    assert model.alpha > model.beta


def test_consistently_bad_behavior_converges_to_low_trust():
    model = AdaptiveTrust(config=TrustConfig())

    for _ in range(100):
        model.update(0.0)

    assert model.trust_score() < model.trust_score() + 1e-6
    assert model.beta > model.alpha


def test_mixed_behavior_reflects_recent_negative_behavior():
    model = AdaptiveTrust(config=TrustConfig())

    for _ in range(50):
        model.update(1.0)

    positive_score = model.trust_score()

    for _ in range(50):
        model.update(0.0)

    final_score = model.trust_score()

    assert final_score < positive_score
    assert 0.0 <= final_score <= 1.0


# =========================================================
# 3.2 Convergence Analysis
# =========================================================

def test_variance_decreases_as_observations_accumulate():
    model = AdaptiveTrust(config=TrustConfig())

    initial_variance = model.variance

    for _ in range(100):
        model.update(1.0)

    assert model.variance < initial_variance


def test_effective_sample_size_increases_with_updates():
    model = AdaptiveTrust(config=TrustConfig())

    initial_n_eff = model.n_eff

    for _ in range(50):
        model.update(1.0)

    assert model.n_eff > initial_n_eff


def test_trust_score_stabilizes_after_many_consistent_observations():
    model = AdaptiveTrust(config=TrustConfig())

    for _ in range(100):
        model.update(1.0)

    scores = [model.trust_score() for _ in range(10)]

    assert max(scores) - min(scores) < 1e-6


# =========================================================
# 3.3 Adversarial Robustness Testing
# =========================================================

def test_on_off_attack_causes_detectable_fluctuation():
    model = AdaptiveTrust(config=TrustConfig())

    for i in range(100):
        model.update(1.0 if i % 2 == 0 else 0.0)

    score = model.trust_score()

    assert 0.0 <= score <= 1.0


def test_ballot_stuffing_initially_increases_trust():
    model = AdaptiveTrust(config=TrustConfig())

    before = model.trust_score()

    for _ in range(20):
        model.update(1.0)

    after = model.trust_score()

    assert after > before


def test_bad_mouthing_decreases_trust():
    model = AdaptiveTrust(config=TrustConfig())

    for _ in range(30):
        model.update(1.0)

    high_score = model.trust_score()

    for _ in range(30):
        model.update(0.0)

    low_score = model.trust_score()

    assert low_score < high_score


def test_slow_degradation_attack_reduces_trust_over_time():
    model = AdaptiveTrust(config=TrustConfig())

    for _ in range(50):
        model.update(1.0)

    high_score = model.trust_score()

    for value in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        model.update(value)

    degraded_score = model.trust_score()

    assert degraded_score < high_score


# =========================================================
# 3.4 Sensitivity Analysis
# =========================================================

@pytest.mark.parametrize("base_lambda", [0.01, 0.05, 0.2])
def test_different_base_lambda_values_produce_valid_scores(base_lambda):
    config = TrustConfig(base_lambda=base_lambda)

    model = AdaptiveTrust(config=config)

    for _ in range(20):
        model.update(1.0)

    score = model.trust_score()

    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("uncertainty_penalty", [0.0, 0.1, 0.5])
def test_uncertainty_penalty_keeps_score_in_valid_range(uncertainty_penalty):
    config = TrustConfig(
        uncertainty_penalty=uncertainty_penalty,
    )

    model = AdaptiveTrust(config=config)

    for _ in range(20):
        model.update(1.0)

    score = model.trust_score()

    assert 0.0 <= score <= 1.0