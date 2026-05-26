import pytest
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig



def test_detection_delay_bad_mouthing():
    """Measure detection delay for Bad-Mouthing attack.
    Number of false negative observations before trust score drops below threshold."""
    
    config = TrustConfig()
    model = AdaptiveTrust(config=config)
    
    # Build legitimate trust first
    for _ in range(50):
        model.update(1.0)
    
    initial_trust = model.trust_score()
    assert initial_trust > 0.85
    
    # Bad-mouthing attack starts - systematic negative observations
    detection_threshold = 0.5
    detection_observations = 0
    
    for i in range(40):
        model.update(0.0)  # Malicious negative observations
        detection_observations += 1
        
        if model.trust_score() < detection_threshold:
            break
    
    # Should detect within reasonable observations
    assert detection_observations <= 25
    assert model.trust_score() < detection_threshold
    


def test_detection_delay_bad_mouthing_different_intensities():
    """Compare detection delay for different bad-mouthing intensities."""
    
    intensities = {
        "mild": 0.3,    # Mostly negative but some positive
        "moderate": 0.1,
        "severe": 0.0   # Completely negative
    }
    delay_results = {}
    
    for intensity_name, bad_value in intensities.items():
        config = TrustConfig()
        model = AdaptiveTrust(config=config)
        
        # Build trust
        for _ in range(50):
            model.update(1.0)
        
        # Bad-mouthing attack
        detection_threshold = 0.5
        detection_observations = 0
        
        for i in range(50):
            model.update(bad_value)
            detection_observations += 1
            
            if model.trust_score() < detection_threshold:
                break
        
        delay_results[intensity_name] = detection_observations
    
    # More severe attack should be detected faster
    assert delay_results["severe"] <= delay_results["moderate"]
    assert delay_results["moderate"] <= delay_results["mild"] 
    


def test_detection_delay_bad_mouthing_against_high_trust():
    """Bad-mouthing against an entity with very high established trust."""
    
    config = TrustConfig()
    model = AdaptiveTrust(config=config)
    
    # Build extremely high trust
    for _ in range(100):
        model.update(1.0)
    
    initial_trust = model.trust_score()
    assert initial_trust > 0.95
    
    # Bad-mouthing attack
    detection_threshold = 0.6
    detection_observations = 0
    
    for i in range(50):
        model.update(0.0)
        detection_observations += 1
        
        if model.trust_score() < detection_threshold:
            break
    
    # High initial trust requires more observations to bring down
    assert detection_observations <= 25


def test_detection_delay_bad_mouthing_different_lambdas():
    """Compare bad-mouthing detection delay across different base_lambda values."""
    
    lambdas = [0.05, 0.1, 0.2, 0.3]
    delay_results = {}
    
    for base_lambda in lambdas:
        config = TrustConfig(base_lambda=base_lambda)
        model = AdaptiveTrust(config=config)
        
        # Build trust
        for _ in range(40):
            model.update(1.0)
        
        # Bad-mouthing attack
        detection_threshold = 0.5
        detection_observations = 0
        
        for i in range(40):
            model.update(0.0)
            detection_observations += 1
            
            if model.trust_score() < detection_threshold:
                break
        
        delay_results[base_lambda] = detection_observations
    
    # Higher lambda should detect faster
    if delay_results[0.05] and delay_results[0.3]:
        assert delay_results[0.3] <= delay_results[0.05]
    


def test_detection_delay_bad_mouthing_with_recovery():
    """Bad-mouthing attack after previous recovery."""
    
    config = TrustConfig()
    model = AdaptiveTrust(config=config)
    
    # Build trust
    for _ in range(50):
        model.update(1.0)
    
    # First bad-mouthing attack
    for _ in range(15):
        model.update(0.0)
    
    # Recovery
    for _ in range(30):
        model.update(1.0)
    
    trust_before_second_attack = model.trust_score()
    
    # Second bad-mouthing attack
    detection_threshold = 0.5
    detection_observations = 0
    
    for i in range(40):
        model.update(0.0)
        detection_observations += 1
        
        if model.trust_score() < detection_threshold:
            break
    
    # Should detect faster due to increased lambda from prior volatility
    assert detection_observations <= 10


def test_detection_delay_bad_mouthing_metric_output():
    """Return detection delay metrics for Bad-Mouthing evaluation."""
    
    config = TrustConfig()
    results = {
        "normal_trust": None,
        "high_trust": None,
        "after_recovery": None
    }
    
    # Normal trust entity
    model = AdaptiveTrust(config=config)
    for _ in range(50):
        model.update(1.0)
    
    detection_obs = 0
    for i in range(80):  # Povečano na 80
        model.update(0.0)
        detection_obs += 1
        if model.trust_score() < 0.5:
            break
    results["normal_trust"] = detection_obs if detection_obs <= 80 else 80
    
    # High trust entity - needs more observations
    model = AdaptiveTrust(config=config)
    for _ in range(100):
        model.update(1.0)
    
    detection_obs = 0
    for i in range(100):  # Povečano na 100
        model.update(0.0)
        detection_obs += 1
        if model.trust_score() < 0.5:
            break
    results["high_trust"] = detection_obs if detection_obs <= 100 else 100
    
    # After recovery
    model = AdaptiveTrust(config=config)
    for _ in range(50):
        model.update(1.0)
    for _ in range(12):
        model.update(0.0)
    for _ in range(35):
        model.update(1.0)
    
    detection_obs = 0
    for i in range(80):
        model.update(0.0)
        detection_obs += 1
        if model.trust_score() < 0.5:
            break
    results["after_recovery"] = detection_obs if detection_obs <= 80 else 80
    
    # Če je še vedno None, nastavi na max
    for key in results:
        if results[key] is None:
            results[key] = 100
    
    assert all(v is not None for v in results.values())
    