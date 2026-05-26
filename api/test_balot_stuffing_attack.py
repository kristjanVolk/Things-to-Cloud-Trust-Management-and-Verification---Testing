import pytest
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig


# =========================================================
# Ballot Stuffing Attack Tests
# =========================================================

def test_ballot_stuffing_increases_trust_for_untrustworthy_entity():
    """Ballot Stuffing: malicious entity submits fraudulently positive observations
    about an untrustworthy peer to inflate its trust score."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    baseline_trust = model.trust_score()
    
    for _ in range(30):
        model.update(1.0)  # Artificially inflated observations
    
    attacked_trust = model.trust_score()
    
    assert attacked_trust > baseline_trust
    assert attacked_trust > 0.95  # Trust should become artificially high
    
    # Despite attack, variance should still reflect some uncertainty
    assert model.variance > 0.0


def test_ballot_stuffing_most_effective_during_early_trust_building():
    """Per specification: Ballot stuffing is especially effective during
    early trust-building phase when n_eff is low."""
    
    # Scenario 1: Attack from the beginning (low n_eff)
    model_early = AdaptiveTrust(config=TrustConfig())
    
    for _ in range(10):
        model_early.update(1.0)  # Fraudulent observations
    
    early_trust = model_early.trust_score()
    
    # Scenario 2: Build legitimate trust first, then attack
    model_late = AdaptiveTrust(config=TrustConfig())
    
    # First build legitimate reputation
    for _ in range(50):
        model_late.update(1.0)
    
    legitimate_trust = model_late.trust_score()
    
    for _ in range(10):
        model_late.update(1.0)
    
    late_attack_trust = model_late.trust_score()
    delta_late = late_attack_trust - legitimate_trust
    
    assert early_trust > 0.90  # Very high trust from few observations
    assert delta_late < 0.01    # Minimal effect after trust is established

    assert model_early.n_eff < model_late.n_eff


def test_ballot_stuffing_detection_delay():
    """Measure detection delay - number of observations before trust
    score correctly reflects malicious behavior."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    
    detection_threshold = 0.7
    observations_needed = 0
    
    for i in range(30):
        model.update(1.0)  # Fraudulent positive observations
        observations_needed += 1
        
        if model.trust_score() >= detection_threshold:
            break
    
    assert observations_needed <= 15
    assert model.trust_score() >= detection_threshold


def test_ballot_stuffing_vs_honest_entity():
    """Compare: Ballot stuffing on untrustworthy vs honest entity behavior."""
    
    malicious_model = AdaptiveTrust(config=TrustConfig())
    for _ in range(20):
        malicious_model.update(1.0)  # Fraudulent
    
    honest_model = AdaptiveTrust(config=TrustConfig())
    for _ in range(20):
        honest_model.update(0.9)  # Real good but not perfect
    
    assert malicious_model.trust_score() > honest_model.trust_score()


def test_ballot_stuffing_uncertainty_penalty_effect():
    """Verify uncertainty penalty doesn't fully prevent ballot stuffing."""
    
    config_high_penalty = TrustConfig(uncertainty_penalty=0.5)
    config_low_penalty = TrustConfig(uncertainty_penalty=0.0)
    
    model_high = AdaptiveTrust(config=config_high_penalty)
    model_low = AdaptiveTrust(config=config_low_penalty)
    
    for _ in range(20):
        model_high.update(1.0)
        model_low.update(1.0)
    
    assert model_low.trust_score() >= model_high.trust_score()
    assert model_high.trust_score() > 0.99  



def test_ballot_stuffing_trust_score_bounds():
    """Ensure trust score stays within valid bounds during ballot stuffing."""
    
    model = AdaptiveTrust(config=TrustConfig())
    
    for _ in range(100):
        model.update(1.0)
    
    score = model.trust_score()
    
    assert 0.0 <= score <= 1.0