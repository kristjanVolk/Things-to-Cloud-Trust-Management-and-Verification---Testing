import pytest
import statistics
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig


class DeterministicTrustModel:
    """Deterministic weighted average with minimum observations."""
    
    def __init__(self, window_size=20, min_observations=10):
        self.window_size = window_size
        self.min_observations = min_observations
        self.observations = []
        self.trust_score_value = 0.5
    
    def update(self, observation):
        self.observations.append(observation)
        if len(self.observations) > self.window_size:
            self.observations.pop(0)
        
        # Only start reporting trust after min_observations
        if len(self.observations) >= self.min_observations:
            self.trust_score_value = sum(self.observations) / len(self.observations)
        else:
            # Conservative estimate with prior
            alpha = 1 + sum(self.observations)
            beta = 1 + len(self.observations) - sum(self.observations)
            self.trust_score_value = alpha / (alpha + beta)
        
        return self.trust_score_value
    
    def trust_score(self):
        return self.trust_score_value


def test_comparative_evaluation_robustness_on_off():
    """Compare robustness against On-Off attack."""
    
    # Stochastic model
    stochastic = AdaptiveTrust(config=TrustConfig())
    
    # Build trust
    for _ in range(40):
        stochastic.update(1.0)
    
    trust_stochastic_before = stochastic.trust_score()
    
    # On-Off attack
    for _ in range(15):
        stochastic.update(0.0)
    
    trust_stochastic_after = stochastic.trust_score()
    stochastic_drop = trust_stochastic_before - trust_stochastic_after
    
    # Deterministic model
    deterministic = DeterministicTrustModel()
    
    for _ in range(40):
        deterministic.update(1.0)
    
    trust_det_before = deterministic.trust_score()
    
    for _ in range(15):
        deterministic.update(0.0)
    
    trust_det_after = deterministic.trust_score()
    det_drop = trust_det_before - trust_det_after
    
    # Both should detect the attack
    assert stochastic_drop > 0.3
    assert det_drop > 0.3
    
    print(f"Stochastic trust drop: {stochastic_drop:.4f}")
    print(f"Deterministic trust drop: {det_drop:.4f}")


def test_comparative_evaluation_robustness_bad_mouthing():
    """Compare robustness against Bad-Mouthing attack."""
    
    # Stochastic model
    stochastic = AdaptiveTrust(config=TrustConfig())
    
    for _ in range(40):
        stochastic.update(1.0)
    
    stochastic_before = stochastic.trust_score()
    
    # Bad-mouthing
    for _ in range(20):
        stochastic.update(0.0)
    
    stochastic_after = stochastic.trust_score()
    stochastic_drop = stochastic_before - stochastic_after
    
    # Deterministic model
    deterministic = DeterministicTrustModel()
    
    for _ in range(40):
        deterministic.update(1.0)
    
    det_before = deterministic.trust_score()
    
    for _ in range(20):
        deterministic.update(0.0)
    
    det_after = deterministic.trust_score()
    det_drop = det_before - det_after
    
    print(f"Stochastic bad-mouthing drop: {stochastic_drop:.4f}")
    print(f"Deterministic bad-mouthing drop: {det_drop:.4f}")
    assert stochastic_drop > 0.3
    assert det_drop > 0.3


def test_comparative_evaluation_convergence_speed():
    """Compare convergence speed between models."""
    
    # Stochastic model convergence
    stochastic = AdaptiveTrust(config=TrustConfig())
    stochastic_obs = 0
    threshold = 0.9
    
    for i in range(1, 101):
        stochastic.update(1.0)
        stochastic_obs += 1
        if stochastic.trust_score() >= threshold:
            break
    
    # Deterministic model convergence
    deterministic = DeterministicTrustModel()
    det_obs = 0
    
    for i in range(1, 101):
        deterministic.update(1.0)
        det_obs += 1
        if deterministic.trust_score() >= threshold:
            break
    
    print(f"Stochastic convergence observations: {stochastic_obs}")
    print(f"Deterministic convergence observations: {det_obs}")
    
    # Both should converge
    assert stochastic_obs <= 80
    assert det_obs <= 80


def test_comparative_evaluation_variance_uncertainty():
    """Compare how models express uncertainty.
    Stochastic model should provide uncertainty metrics that deterministic lacks."""
    
    stochastic = AdaptiveTrust(config=TrustConfig())
    
    # Early stage (high uncertainty)
    for _ in range(5):
        stochastic.update(1.0)
    
    early_variance = stochastic.variance
    early_n_eff = stochastic.n_eff
    
    # Late stage (low uncertainty)
    for _ in range(95):
        stochastic.update(1.0)
    
    late_variance = stochastic.variance
    late_n_eff = stochastic.n_eff
    
    # Stochastic model shows decreasing uncertainty
    assert late_variance < early_variance
    assert late_n_eff > early_n_eff
    
    # Deterministic model has no uncertainty metric
    deterministic = DeterministicTrustModel()
    assert not hasattr(deterministic, 'variance')
    
    print(f"Stochastic uncertainty decreases: {early_variance:.6f} -> {late_variance:.6f}")
    print(f"Stochastic n_eff increases: {early_n_eff:.2f} -> {late_n_eff:.2f}")


# ...existing code...
def test_comparative_evaluation_under_volatility():
    """Compare model behavior under volatile conditions."""
    
    # Stochastic model
    stochastic = AdaptiveTrust(config=TrustConfig())
    
    # Deterministic model
    deterministic = DeterministicTrustModel()
    
    # Volatile behavior
    volatile_sequence = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] * 10
    
    stochastic_scores = []
    det_scores = []
    
    for obs in volatile_sequence:
        stochastic.update(obs)
        deterministic.update(obs)
        stochastic_scores.append(stochastic.trust_score())
        det_scores.append(deterministic.trust_score())
    
    stochastic_variance = statistics.variance(stochastic_scores[-30:])
    det_variance = statistics.variance(det_scores[-30:])
    
    print(f"Stochastic trust variance (volatile): {stochastic_variance:.6f}")
    print(f"Deterministic trust variance (volatile): {det_variance:.6f}")
    
    assert det_variance <= stochastic_variance


def test_comparative_evaluation_metrics_summary():
    
    results = {
        "stochastic": {},
        "deterministic": {}
    }
    
    # Test 1: Accuracy on mixed behavior (70% good, 30% bad)
    stochastic = AdaptiveTrust(config=TrustConfig())
    deterministic = DeterministicTrustModel()
    
    observations = [1.0] * 70 + [0.0] * 30
    
    for obs in observations:
        stochastic.update(obs)
        deterministic.update(obs)
    
    results["stochastic"]["mixed_accuracy"] = abs(stochastic.trust_score() - 0.7)
    results["deterministic"]["mixed_accuracy"] = abs(deterministic.trust_score() - 0.7)
    
    # Test 2: Attack recovery
    stochastic = AdaptiveTrust(config=TrustConfig())
    deterministic = DeterministicTrustModel()
    
    for _ in range(40):
        stochastic.update(1.0)
        deterministic.update(1.0)
    
    for _ in range(15):
        stochastic.update(0.0)
        deterministic.update(0.0)
    
    for _ in range(30):
        stochastic.update(1.0)
        deterministic.update(1.0)
    
    results["stochastic"]["recovery"] = stochastic.trust_score()
    results["deterministic"]["recovery"] = deterministic.trust_score()
    
    # Print summary
    print("\n" + "="*50)
    print("COMPARATIVE EVALUATION SUMMARY (Section 3.5)")
    print("="*50)
    print(f"Mixed Behavior Accuracy (error vs 0.7):")
    print(f"  Stochastic:    {results['stochastic']['mixed_accuracy']:.4f}")
    print(f"  Deterministic: {results['deterministic']['mixed_accuracy']:.4f}")
    print(f"\nRecovery Trust Score after attack:")
    print(f"  Stochastic:    {results['stochastic']['recovery']:.4f}")
    print(f"  Deterministic: {results['deterministic']['recovery']:.4f}")
    print("="*50)
    
    assert results["stochastic"]["recovery"] > 0.7

def test_measure_deterministic_convergence():
    """Dejansko izmeri konvergenco deterministic modela."""
    from test_comperative_evaluation import DeterministicTrustModel
    
    model = DeterministicTrustModel(window_size=20)
    for i in range(1, 101):
        model.update(1.0)
        if model.trust_score() >= 0.9:
            print(f"Deterministic converged at: {i} observations")
            break
    assert True