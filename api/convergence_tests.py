import pytest
import statistics
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig


# =========================================================
# 3.2 Convergence Analysis Tests
# =========================================================

def test_variance_decreases_as_observations_accumulate():
    """Per specification: variance should decrease as more evidence
    accumulates, reflecting higher confidence in trust estimate."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    initial_variance = model.variance
    variances = [initial_variance]
    
    # Update with consistent good behavior
    for i in range(1, 101):
        model.update(1.0)
        if i % 10 == 0:
            variances.append(model.variance)
    
    # Variance should generally decrease over time
    for i in range(1, len(variances)):
        assert variances[i] <= variances[i-1] * 1.1  # Allow small fluctuations
    
    final_variance = model.variance
    assert final_variance < initial_variance
    assert final_variance < 0.001  # Very low uncertainty after 100 observations



def test_trust_score_stabilizes_within_95_percent_confidence_interval():
    """Per specification: measure number of observations required for
    trust score to reach stable value within 95% confidence interval.
    """
    
    model = AdaptiveTrust(config=TrustConfig())
    scores = []
    
    # Record trust scores over time
    for i in range(200):
        model.update(1.0)
        scores.append(model.trust_score())
    
    # Calculate rolling mean and confidence interval
    window_size = 20
    stabilization_observation = None
    
    for i in range(window_size, len(scores) - window_size):
        window = scores[i-window_size:i+window_size]
        mean = statistics.mean(window)
        stdev = statistics.stdev(window) if len(window) > 1 else 0
        ci_95 = 1.96 * stdev  # 95% confidence interval
        
        # Stabilized if confidence interval is very small
        if ci_95 < 0.01:
            stabilization_observation = i
            break
    
    # Should stabilize within reasonable number of observations
    assert stabilization_observation is not None
    assert stabilization_observation <= 34
    assert scores[-1] > 0.95  


def test_convergence_speed_for_good_behavior():
    """Measure how quickly trust converges to 0.95 for consistently
    good behavior."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    convergence_threshold = 0.95
    observations_needed = 0
    
    for i in range(1, 101):
        model.update(1.0)
        observations_needed += 1
        
        if model.trust_score() >= convergence_threshold:
            break
    
    # Should converge to high trust within reasonable observations
    assert observations_needed <= 20
    assert model.trust_score() >= convergence_threshold


def test_convergence_speed_for_bad_behavior():
    """Measure how quickly trust converges to 0.05 for consistently
    bad behavior."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    convergence_threshold = 0.05
    observations_needed = 0
    
    for i in range(1, 101):
        model.update(0.0)
        observations_needed += 1
        
        if model.trust_score() <= convergence_threshold:
            break
    
    # Should converge to low trust within reasonable observations
    assert observations_needed <= 15
    assert model.trust_score() <= convergence_threshold


def test_uncertainty_decreases_with_effective_sample_size():
    """Verify that uncertainty (variance) is inversely related to n_eff."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    variance_n_eff_pairs = []
    
    for i in range(100):
        model.update(1.0)
        if i % 10 == 0:
            variance_n_eff_pairs.append((model.variance, model.n_eff))
    
    # As n_eff increases, variance should decrease
    for i in range(1, len(variance_n_eff_pairs)):
        prev_variance, prev_n_eff = variance_n_eff_pairs[i-1]
        curr_variance, curr_n_eff = variance_n_eff_pairs[i]
        
        if curr_n_eff > prev_n_eff:
            assert curr_variance <= prev_variance


def test_convergence_rate_comparison_different_lambdas():
    """Compare convergence speed for different base_lambda values.
    Higher lambda should converge faster but may be less stable."""
    
    config_fast = TrustConfig(base_lambda=0.3)
    config_slow = TrustConfig(base_lambda=0.05)
    
    model_fast = AdaptiveTrust(config=config_fast)
    model_slow = AdaptiveTrust(config=config_slow)
    
    # Track observations needed to reach threshold
    threshold = 0.9
    fast_obs = 0
    slow_obs = 0
    
    # Fast lambda model
    for i in range(1, 101):
        model_fast.update(1.0)
        fast_obs += 1
        if model_fast.trust_score() >= threshold:
            break
    
    # Slow lambda model
    for i in range(1, 201):
        model_slow.update(1.0)
        slow_obs += 1
        if model_slow.trust_score() >= threshold:
            break
    
    # Fast should converge faster (or equal)
    print(f"fast obs: {fast_obs}, slow obs: {slow_obs}")
    assert fast_obs <= slow_obs
    assert fast_obs <= 10  # Fast should be quick
    assert slow_obs <= 40  # Slow takes longer but still converges


def test_stabilization_after_behavior_change():
    """Test convergence when entity behavior changes mid-stream.
    Trust should adapt and converge to new behavior pattern."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    # Phase 1: Good behavior
    for _ in range(50):
        model.update(1.0)
    
    trust_during_good = model.trust_score()
    assert trust_during_good > 0.85
    
    # Phase 2: Behavior changes to bad
    for _ in range(80):
        model.update(0.0)
    
    trust_after_bad = model.trust_score()
    
    # Should converge toward bad behavior

    assert trust_after_bad < 0.3
    # Variance should initially increase during transition, then decrease
    assert model.variance < 0.05  

def test_convergence_95ci_stabilization_metric():
    """Direct implementation of the convergence metric from section 3.2:
    'number of observations required for the trust score to reach a stable
    value within a 95% confidence interval'."""
    
    model = AdaptiveTrust(config=TrustConfig())
    trust_history = []
    
    # Collect trust scores
    for i in range(200):
        model.update(1.0)
        trust_history.append(model.trust_score())
    
    # Find stabilization point using 95% CI with rolling window
    window = 30
    stabilized = False
    stabilization_index = len(trust_history)
    
    for i in range(window, len(trust_history) - window):
        window_values = trust_history[i-window:i+window]
        mean = statistics.mean(window_values)
        stdev = statistics.stdev(window_values)
        ci_95_width = 1.96 * stdev * 2  # Full width of CI
        
        # Stabilized when 95% CI width is less than 0.02
        if ci_95_width < 0.02:
            stabilization_index = i
            stabilized = True
            break
    
    assert stabilization_index <= 50
    assert model.variance < 0.001


def test_convergence_with_uncertainty_penalty():
    """Uncertainty penalty should not prevent convergence,
    but may affect the final trust value."""
    
    config_no_penalty = TrustConfig(uncertainty_penalty=0.0)
    config_with_penalty = TrustConfig(uncertainty_penalty=0.3)
    
    model_no_penalty = AdaptiveTrust(config=config_no_penalty)
    model_with_penalty = AdaptiveTrust(config=config_with_penalty)
    
    # Both models get same good observations
    for _ in range(100):
        model_no_penalty.update(1.0)
        model_with_penalty.update(1.0)
    
    # Both should converge to high trust
    assert model_no_penalty.trust_score() > 0.9
    assert model_with_penalty.trust_score() > 0.85
    
    # Penalty model may have slightly lower trust due to uncertainty adjustment
    # But both should be stable
    assert model_no_penalty.variance < 0.01
    assert model_with_penalty.variance < 0.01



