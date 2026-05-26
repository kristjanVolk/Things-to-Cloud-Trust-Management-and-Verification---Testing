import pytest
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig


# =========================================================
# Slow Degradation Attack Tests
# =========================================================

def test_slow_degradation_gradually_reduces_trust():
    """Slow Degradation: entity with high trust gradually reduces performance."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    # Build high trust first
    for _ in range(50):
        model.update(1.0)
    
    initial_high_trust = model.trust_score()
    assert initial_high_trust > 0.9
    
    # Slow degradation - longer sequence
    degradation_values = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
    
    scores_over_time = []
    for value in degradation_values:
        model.update(value)
        scores_over_time.append(model.trust_score())
    
    final_trust = model.trust_score()
    
    # Trust should decrease over time
    assert final_trust < initial_high_trust
    assert final_trust < 0.5
    assert scores_over_time[-1] < scores_over_time[0]


def test_slow_degradation_below_volatility_threshold():
    """Critical test: degradation so slow that volatility stays low."""
    
    config = TrustConfig(window_size=10)
    model = AdaptiveTrust(config=config)
    
    # Build high trust
    for _ in range(50):
        model.update(1.0)
    
    # Extremely slow degradation (0.5% decrease per observation)
    current_value = 1.0
    trust_scores = []
    volatilities = []
    
    for i in range(80):
        current_value = max(0.4, current_value - 0.005)
        model.update(current_value)
        if i % 10 == 0:
            trust_scores.append(model.trust_score())
            volatilities.append(model.compute_volatility())
    
    # Volatility should remain low
    final_volatility = model.compute_volatility()
    
    # Trust should decrease despite low volatility
    assert model.trust_score() < 0.7
    assert final_volatility < 0.10


def test_slow_degradation_detection_delay():
    """Measure detection delay - how many observations before trust reflects degradation."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    # Build high trust
    for _ in range(50):
        model.update(1.0)
    
    degradation_threshold = 0.75
    observations_to_degradation = 0
    
    # Slow degradation
    current_value = 1.0
    for i in range(80):
        current_value = max(0.3, current_value - 0.01)
        model.update(current_value)
        observations_to_degradation += 1
        
        if model.trust_score() < degradation_threshold:
            break
    
    assert observations_to_degradation > 10
    assert observations_to_degradation < 30
    assert model.trust_score() < degradation_threshold


def test_slow_degradation_vs_sudden_drop():
    """Compare slow degradation vs sudden drop - both should reduce trust."""
    
    # Slow degradation model
    slow_model = AdaptiveTrust(config=TrustConfig())
    for _ in range(50):
        slow_model.update(1.0)
    
    # Longer and deeper slow degradation
    for v in [0.94, 0.88, 0.82, 0.76, 0.70, 0.64, 0.58, 0.52, 0.46, 0.40]:
        slow_model.update(v)
    
    slow_final = slow_model.trust_score()
    
    # Sudden drop model
    sudden_model = AdaptiveTrust(config=TrustConfig())
    for _ in range(50):
        sudden_model.update(1.0)
    
    # Sudden drop to 0.4
    sudden_model.update(0.4)
    sudden_final = sudden_model.trust_score()
    
    # Both should reduce trust below 0.85
    assert slow_final < 0.85
    assert sudden_final < 0.85
    
    # Volatility should be higher for sudden drop
    assert sudden_model.compute_volatility() > slow_model.compute_volatility()


def test_slow_degradation_recovery():
    """After slow degradation, can the entity recover trust?"""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    # Build high trust
    for _ in range(50):
        model.update(1.0)
    
    peak_trust = model.trust_score()
    
    # Slow degradation phase - longer
    for v in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]:
        model.update(v)
    
    degraded_trust = model.trust_score()
    
    # Recovery phase - return to good behavior
    for _ in range(50):
        model.update(1.0)
    
    recovered_trust = model.trust_score()
    
    # Recovery should happen but may not reach original peak
    assert degraded_trust < peak_trust
    assert recovered_trust > degraded_trust
    


def test_slow_degradation_with_fluctuations():
    """Slow degradation with small fluctuations - should still be detectable."""
    
    model = AdaptiveTrust(config=TrustConfig(window_size=8))
    
    # Build trust
    for _ in range(40):
        model.update(1.0)
    
    initial_trust = model.trust_score()
    
    # Degradation with noise
    current_base = 1.0
    for i in range(60):
        current_base = max(0.4, current_base - 0.01)
        # Add small noise
        noisy_value = current_base + (i % 10) / 100.0 - 0.05
        noisy_value = max(0.0, min(1.0, noisy_value))
        model.update(noisy_value)
    
    final_trust = model.trust_score()
    
    assert final_trust < initial_trust
    assert final_trust < 0.5


def test_slow_degradation_trust_score_bounds():
    """Ensure trust score stays within valid bounds during slow degradation."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    # Build trust
    for _ in range(30):
        model.update(1.0)
    
    # Degrade all the way to 0
    current = 1.0
    for _ in range(100):
        current = max(0.0, current - 0.01)
        model.update(current)
        score = model.trust_score()
        assert 0.0 <= score <= 1.0
    
    final_score = model.trust_score()
    assert 0.0 <= final_score <= 1.0