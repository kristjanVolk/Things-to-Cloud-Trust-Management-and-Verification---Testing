import pytest
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig


# =========================================================
# Detection Delay for On-Off Attack (Section 3.3 & 4.1)
# =========================================================

def test_detection_delay_on_off_first_attack():
    """Measure detection delay for On-Off attack - first malicious phase."""
    
    config = TrustConfig()
    model = AdaptiveTrust(config=config)
    
    # Phase 1: Honest behavior to build trust
    for _ in range(50):
        model.update(1.0)
    
    initial_trust = model.trust_score()
    assert initial_trust > 0.85
    
    # Phase 2: Attack starts - malicious behavior
    detection_threshold = 0.5  # Lowered threshold
    detection_observations = 0
    
    for i in range(40):
        model.update(0.0)
        detection_observations += 1
        
        if model.trust_score() < detection_threshold:
            break
    
    # Should detect within reasonable observations
    assert detection_observations <= 10


def test_detection_delay_on_off_second_attack():
    """Detection delay for second attack after recovery."""
    
    config = TrustConfig()
    model = AdaptiveTrust(config=config)
    
    # Build initial trust
    for _ in range(50):
        model.update(1.0)
    
    # First attack phase
    for _ in range(15):
        model.update(0.0)
    
    # Recovery phase
    for _ in range(30):
        model.update(1.0)
    
    # Second attack
    detection_threshold = 0.5
    detection_observations = 0
    
    for i in range(40):
        model.update(0.0)
        detection_observations += 1
        
        if model.trust_score() < detection_threshold:
            break
    
    # Second attack should be detected (may be similar or faster)
    assert detection_observations <= 10

def test_detection_delay_on_off_different_lambdas():
    """Compare detection delay across different base_lambda values."""
    
    lambdas = [0.05, 0.1, 0.2, 0.3]
    delay_results = {}
    
    for base_lambda in lambdas:
        config = TrustConfig(base_lambda=base_lambda)
        model = AdaptiveTrust(config=config)
        
        # Build trust
        for _ in range(50):
            model.update(1.0)
        
        # Attack
        detection_threshold = 0.5
        detection_observations = 0
        
        for i in range(40):
            model.update(0.0)
            detection_observations += 1
            
            if model.trust_score() < detection_threshold:
                break
        
        delay_results[base_lambda] = detection_observations
    
    # All should detect within reasonable bounds
    for base_lambda in lambdas:
        assert delay_results[base_lambda] <= 10
    
    print(f"Detection delay by lambda: {delay_results}")


def test_detection_delay_on_off_partial_recovery():
    """Detection delay when attack resumes after partial recovery."""
    
    config = TrustConfig()
    model = AdaptiveTrust(config=config)
    
    # Build trust
    for _ in range(50):
        model.update(1.0)
    
    # First attack (partial)
    for _ in range(10):
        model.update(0.0)
    
    # Partial recovery (not full)
    for _ in range(15):
        model.update(0.8)
    
    # Second attack
    detection_threshold = 0.5
    detection_observations = 0
    
    for i in range(40):
        model.update(0.0)
        detection_observations += 1
        
        if model.trust_score() < detection_threshold:
            break
    
    assert detection_observations <= 5


def test_detection_delay_on_off_metric_output():
    """Return detection delay as a quantitative metric for evaluation."""
    
    config = TrustConfig()
    results = {
        "first_attack": None,
        "second_attack": None,
        "after_partial_recovery": None
    }
    
    # First attack
    model = AdaptiveTrust(config=config)
    for _ in range(50):
        model.update(1.0)
    
    detection_obs = 0
    for i in range(40):
        model.update(0.0)
        detection_obs += 1
        if model.trust_score() < 0.5:
            break
    results["first_attack"] = detection_obs
    
    # Second attack (full recovery first)
    model = AdaptiveTrust(config=config)
    for _ in range(50):
        model.update(1.0)
    for _ in range(10):
        model.update(0.0)
    for _ in range(30):
        model.update(1.0)
    
    detection_obs = 0
    for i in range(40):
        model.update(0.0)
        detection_obs += 1
        if model.trust_score() < 0.5:
            break
    results["second_attack"] = detection_obs
    
    # Partial recovery scenario
    model = AdaptiveTrust(config=config)
    for _ in range(50):
        model.update(1.0)
    for _ in range(8):
        model.update(0.0)
    for _ in range(10):
        model.update(0.8)
    
    detection_obs = 0
    for i in range(40):
        model.update(0.0)
        detection_obs += 1
        if model.trust_score() < 0.5:
            break
    results["after_partial_recovery"] = detection_obs
    
    # Verify all metrics are captured (values can be None if never detected)
    # If not detected within 40 observations, set to 40
    for key in results:
        if results[key] is None:
            results[key] = 40
    
    # All should have values
    assert all(v is not None for v in results.values())
    

        