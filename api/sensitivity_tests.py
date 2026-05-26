import pytest
import statistics
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig


# =========================================================
# 3.4 Sensitivity Analysis Tests
# =========================================================

def test_base_lambda_effect_on_convergence_speed():
    """Per specification: vary base_lambda while holding others constant.
    Higher base_lambda should lead to faster convergence but potentially
    more volatility."""
    
    lambdas = [0.05, 0.1, 0.2, 0.3, 0.5]
    convergence_results = {}
    
    for base_lambda in lambdas:
        config = TrustConfig(
            base_lambda=base_lambda,
            uncertainty_penalty=0.1
        )
        model = AdaptiveTrust(config=config)
        
        # Measure observations to reach threshold
        threshold = 0.9
        observations = 0
        
        for i in range(1, 101):
            model.update(1.0)
            observations += 1
            if model.trust_score() >= threshold:
                break
        
        convergence_results[base_lambda] = observations
    
    # Higher lambda should converge faster (fewer observations)
    # Check general trend
    if convergence_results[0.05] and convergence_results[0.5]:
        assert convergence_results[0.5] <= convergence_results[0.05] * 1.5


def test_base_lambda_effect_on_stability():
    """Higher base_lambda may cause more fluctuations in trust score."""
    
    # Low lambda model (stable)
    config_low = TrustConfig(base_lambda=0.05)
    model_low = AdaptiveTrust(config=config_low)
    
    # High lambda model (sensitive)
    config_high = TrustConfig(base_lambda=0.5)
    model_high = AdaptiveTrust(config=config_high)
    
    # Build trust
    for _ in range(50):
        model_low.update(1.0)
        model_high.update(1.0)
    
    trust_low_before = model_low.trust_score()
    trust_high_before = model_high.trust_score()
    
    # Single bad observation
    model_low.update(0.0)
    model_high.update(0.0)
    
    trust_low_after = model_low.trust_score()
    trust_high_after = model_high.trust_score()
    
    drop_low = trust_low_before - trust_low_after
    drop_high = trust_high_before - trust_high_after
    
    # Higher lambda should cause larger drop (more sensitivity)
    assert drop_high >= drop_low  # Allow small tolerance



def test_T_stable_effect_on_volatility():
    """T_stable controls the window for volatility calculation.
    Larger T_stable means slower adaptation to behavioral changes."""
    
    # Small T_stable (more sensitive to recent changes)
    config_small = TrustConfig(T_stable=50.0, window_size=10)
    model_small = AdaptiveTrust(config=config_small)
    
    # Large T_stable (less sensitive)
    config_large = TrustConfig(T_stable=200.0, window_size=10)
    model_large = AdaptiveTrust(config=config_large)
    
    # Build trust
    for _ in range(30):
        model_small.update(1.0)
        model_large.update(1.0)
    
    trust_small_before = model_small.trust_score()
    trust_large_before = model_large.trust_score()
    
    # Sudden behavior change
    for _ in range(15):
        model_small.update(0.0)
        model_large.update(0.0)
    
    trust_small_after = model_small.trust_score()
    trust_large_after = model_large.trust_score()
    
    drop_small = trust_small_before - trust_small_after
    drop_large = trust_large_before - trust_large_after
    
    # Smaller T_stable should adapt faster (larger drop)
    assert drop_small > drop_large


def test_uncertainty_penalty_effect_on_trust_score():
    """Per specification: uncertainty_penalty penalizes uncertain trust
    estimates. Higher penalty should result in lower trust scores when
    uncertainty is high."""
    
    penalties = [0.0, 0.1, 0.3, 0.5]
    trust_scores_early = {}
    trust_scores_late = {}
    
    for penalty in penalties:
        config = TrustConfig(
            base_lambda=0.2,
            uncertainty_penalty=penalty
        )
        model = AdaptiveTrust(config=config)
        
        # Early stage (high uncertainty)
        for _ in range(5):
            model.update(1.0)
        trust_scores_early[penalty] = model.adjusted_trust_score()
        
        # Late stage (low uncertainty)
        for _ in range(95):
            model.update(1.0)
        trust_scores_late[penalty] = model.adjusted_trust_score()
    
    # Higher penalty should reduce trust score more in early stage (high uncertainty)
    for i in range(1, len(penalties)):
        if trust_scores_early[penalties[i]] and trust_scores_early[penalties[i-1]]:
            # Higher penalty -> lower or equal trust
            assert trust_scores_early[penalties[i]] <= trust_scores_early[penalties[i-1]] + 0.05
    
    # Late stage should have minimal penalty effect (low uncertainty)
    late_variation = max(trust_scores_late.values()) - min(trust_scores_late.values())
    assert late_variation < 0.1  # 


def test_uncertainty_penalty_effect_on_volatile_behavior():
    """Uncertainty penalty should have stronger effect when behavior
    is volatile (high variance)."""
    
    config_no_penalty = TrustConfig(uncertainty_penalty=0.0)
    config_high_penalty = TrustConfig(uncertainty_penalty=0.5)
    
    model_no_penalty = AdaptiveTrust(config=config_no_penalty)
    model_high_penalty = AdaptiveTrust(config=config_high_penalty)
    
    # Simulate volatile behavior (alternating good and bad)
    for i in range(50):
        value = 1.0 if i % 2 == 0 else 0.0
        model_no_penalty.update(value)
        model_high_penalty.update(value)
    
    # High penalty should result in lower trust due to uncertainty
    assert model_high_penalty.adjusted_trust_score() <= model_no_penalty.adjusted_trust_score()
    
    # Variance should be similar (penalty doesn't change variance)
    assert model_high_penalty.variance == pytest.approx(model_no_penalty.variance)


def test_window_size_effect_on_volatility_detection():
    """window_size controls how many observations are used for volatility.
    Larger window = smoother but slower to detect changes."""
    
    # Small window (reactive)
    config_small = TrustConfig(window_size=3)
    model_small = AdaptiveTrust(config=config_small)
    
    # Large window (stable)
    config_large = TrustConfig(window_size=20)
    model_large = AdaptiveTrust(config=config_large)
    
    # Build stable history
    for _ in range(30):
        model_small.update(1.0)
        model_large.update(1.0)
    
    # Introduce volatility
    for _ in range(10):
        model_small.update(0.0)
        model_small.update(1.0)
        model_large.update(0.0)
        model_large.update(1.0)
    
    # Small window should show higher volatility (more reactive)
    vol_small = model_small.compute_volatility()
    vol_large = model_large.compute_volatility()
    
    assert vol_small >= vol_large


def test_parameter_interaction_base_lambda_and_uncertainty_penalty():
    """Test interaction between base_lambda and uncertainty_penalty."""
    
    # Fast adaptation with low penalty
    config_fast = TrustConfig(base_lambda=0.4, uncertainty_penalty=0.0)
    model_fast = AdaptiveTrust(config=config_fast)
    
    # Slow adaptation with high penalty
    config_slow = TrustConfig(base_lambda=0.05, uncertainty_penalty=0.5)
    model_slow = AdaptiveTrust(config=config_slow)
    
    # Build trust
    for _ in range(30):
        model_fast.update(1.0)
        model_slow.update(1.0)
    
    trust_fast_before = model_fast.trust_score()
    trust_slow_before = model_slow.trust_score()
    
    # Sudden behavior change
    for _ in range(20):
        model_fast.update(0.0)
        model_slow.update(0.0)
    
    trust_fast_after = model_fast.trust_score()
    trust_slow_after = model_slow.trust_score()
    
    drop_fast = trust_fast_before - trust_fast_after
    drop_slow = trust_slow_before - trust_slow_after
    
    # Fast configuration should adapt more quickly (larger drop)
    assert drop_fast > drop_slow


def test_parameter_sweep_all_combinations():
    """Systematic parameter sweep across all key parameters.
    This test ensures all combinations produce valid trust scores."""
    
    base_lambdas = [0.05, 0.1, 0.2, 0.4]
    uncertainty_penalties = [0.0, 0.2, 0.5]
    window_sizes = [5, 10, 20]
    
    for base_lambda in base_lambdas:
        for penalty in uncertainty_penalties:
            for window_size in window_sizes:
                config = TrustConfig(
                    base_lambda=base_lambda,
                    uncertainty_penalty=penalty,
                    window_size=window_size
                )
                model = AdaptiveTrust(config=config)
                
                # Run simulation
                for _ in range(50):
                    model.update(0.8)
                
                # Verify outputs are valid
                assert 0.0 <= model.trust_score() <= 1.0
                assert 0.0 <= model.adjusted_trust_score() <= 1.0
                assert model.variance > 0.0
                assert model.n_eff > 0.0
                assert 0.0 <= model.compute_lambda() <= 1.0


def test_sensitivity_metric_detection_delay_vs_lambda():
    """Measure how base_lambda affects detection delay for attacks.
    Lower lambda = slower detection but more stable.
    Higher lambda = faster detection but potentially more false positives."""
    
    lambdas = [0.05, 0.1, 0.2, 0.3]
    detection_delays = {}
    
    for base_lambda in lambdas:
        config = TrustConfig(base_lambda=base_lambda)
        model = AdaptiveTrust(config=config)
        
        # Build trust
        for _ in range(40):
            model.update(1.0)
        
        # Attack starts
        observations_to_detect = 0
        initial_trust = model.trust_score()
        
        for i in range(30):
            model.update(0.0)
            observations_to_detect += 1
            
            # Detection when trust drops by 30% from peak
            if model.trust_score() < initial_trust * 0.7:
                break
        
        detection_delays[base_lambda] = observations_to_detect
    
    # Higher lambda should detect faster (fewer observations)
    if detection_delays[0.05] and detection_delays[0.3]:
        assert detection_delays[0.3] <= detection_delays[0.05] * 1.5


def test_sensitivity_metric_false_positive_rate_vs_lambda():
    """Measure how base_lambda affects false positive rate.
    Higher lambda may cause more false positives on honest entities
    with normal fluctuations."""
    
    lambdas = [0.05, 0.1, 0.2, 0.3, 0.5]
    false_positives = {}
    
    for base_lambda in lambdas:
        config = TrustConfig(base_lambda=base_lambda)
        model = AdaptiveTrust(config=config)
        
        # Simulate honest entity with normal small fluctuations
        false_positive_count = 0
        threshold = 0.5  # Below this is considered "untrustworthy"
        
        for i in range(100):
            # Honest behavior: mostly good with occasional small dips
            if i % 20 == 0:
                value = 0.7  # Small dip
            else:
                value = 0.95
            
            model.update(value)
            
            if model.trust_score() < threshold and i > 20:  # After initial buildup
                false_positive_count += 1
        
        false_positives[base_lambda] = false_positive_count
    
    # Both should have low false positives for honest entities
    for base_lambda in lambdas:
        assert false_positives[base_lambda] <= 10


def test_parameter_recommendation_range():
    """Identify parameter ranges that provide good balance between
    convergence speed and stability."""
    
    # Test recommended configuration (balanced)
    config_balanced = TrustConfig(
        base_lambda=0.15,
        uncertainty_penalty=0.15,
        window_size=10,
        n_target=20.0
    )
    
    model = AdaptiveTrust(config=config_balanced)
    
    # Measure both convergence speed and stability
    convergence_obs = 0
    trust_history = []
    
    for i in range(100):
        model.update(1.0)
        convergence_obs += 1
        trust_history.append(model.trust_score())
        
        if model.trust_score() >= 0.9:
            break
    
    # Should converge within reasonable time
    assert convergence_obs <= 70
    
    # Should be stable (low variance in last 20 observations)
    last_20 = trust_history[-20:] if len(trust_history) >= 20 else trust_history
    if len(last_20) > 1:
        stability_variance = statistics.variance(last_20)
        assert stability_variance < 0.015