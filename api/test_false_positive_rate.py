import pytest
import statistics
from app.services.trust.adaptive_trust import AdaptiveTrust
from app.services.trust.trust_config import TrustConfig


# =========================================================
# False Positive Rate Metrics (Section 3.3 & 4.1)
# =========================================================

def test_false_positive_rate_for_honest_entities():
    """Measure false positive rate - honest entities incorrectly penalized.
    Per specification: 'false positive rate — the rate at which honest 
    entities are incorrectly penalized'"""
    
    config = TrustConfig()
    threshold = 0.3  # Below this is considered "penalized/excluded"
    
    false_positives = 0
    total_evaluations = 0
    
    # Simulate multiple honest entities
    for entity in range(10):
        model = AdaptiveTrust(config=config)
        
        # Each entity has normal honest behavior with small fluctuations
        for i in range(100):
            # Normal honest behavior: 0.85-1.0 range with small dips
            if i % 25 == 0:
                value = 0.75  # Occasional minor dip (still honest)
            else:
                value = 0.92 + (i % 10) / 100.0 - 0.05
            
            model.update(value)
            total_evaluations += 1
            
            if model.trust_score() < threshold and i > 20:  # After initial buildup
                false_positives += 1
    
    fpr = false_positives / total_evaluations if total_evaluations > 0 else 1.0
    assert fpr < 0.05  # Less than 5% false positive rate


def test_false_positive_rate_different_thresholds():
    """Measure FPR for different acceptance thresholds."""
    
    thresholds = [0.2, 0.3, 0.4, 0.5]
    fpr_results = {}
    
    for threshold in thresholds:
        config = TrustConfig()
        false_positives = 0
        total_evaluations = 0
        
        for entity in range(5):
            model = AdaptiveTrust(config=config)
            
            for i in range(100):
                # Honest behavior
                value = 0.9 + (i % 8) / 100.0
                model.update(value)
                total_evaluations += 1
                
                if model.trust_score() < threshold and i > 20:
                    false_positives += 1
        
        fpr_results[threshold] = false_positives / total_evaluations if total_evaluations > 0 else 1.0
    
    # Higher threshold should result in higher FPR
    for i in range(1, len(thresholds)):
        assert fpr_results[thresholds[i]] >= fpr_results[thresholds[i-1]]


def test_false_positive_rate_with_noise():
    """Honest entities with legitimate noise should not trigger false positives."""
    
    config = TrustConfig()
    threshold = 0.3
    false_positives = 0
    total_evaluations = 0
    
    for entity in range(5):
        model = AdaptiveTrust(config=config)
        
        for i in range(100):
            # Honest behavior with legitimate noise
            import random
            random.seed(entity * 100 + i)  # Reproducible
            noise = random.uniform(-0.05, 0.05)
            value = max(0.8, min(1.0, 0.9 + noise))
            model.update(value)
            total_evaluations += 1
            
            if model.trust_score() < threshold and i > 20:
                false_positives += 1
    
    fpr = false_positives / total_evaluations if total_evaluations > 0 else 1.0
    assert fpr < 0.03  # Very low FPR for legitimate noise
    


def test_false_positive_rate_different_lambdas():
    """Compare FPR across different base_lambda values.
    Higher lambda may cause more false positives."""
    
    lambdas = [0.05, 0.1, 0.2, 0.3, 0.5]
    fpr_results = {}
    threshold = 0.3
    
    for base_lambda in lambdas:
        config = TrustConfig(base_lambda=base_lambda)
        false_positives = 0
        total_evaluations = 0
        
        for entity in range(5):
            model = AdaptiveTrust(config=config)
            
            for i in range(100):
                # Honest behavior with small dips
                if i % 30 == 0:
                    value = 0.8
                else:
                    value = 0.95
                
                model.update(value)
                total_evaluations += 1
                
                if model.trust_score() < threshold and i > 30:
                    false_positives += 1
        
        fpr_results[base_lambda] = false_positives / total_evaluations if total_evaluations > 0 else 1.0
    
    # All should have reasonable FPR
    for base_lambda in lambdas:
        assert fpr_results[base_lambda] < 0.1
    


def test_false_positive_rate_after_convergence():
    """Once trust is established, FPR should be near zero."""
    
    config = TrustConfig()
    threshold = 0.3
    
    for entity in range(5):
        model = AdaptiveTrust(config=config)
        
        # Build strong trust first
        for _ in range(50):
            model.update(1.0)
        
        assert model.trust_score() > 0.85
        
        # Continue with honest behavior
        false_positives = 0
        for i in range(100):
            value = 0.92 + (i % 10) / 100.0
            model.update(value)
            
            if model.trust_score() < threshold:
                false_positives += 1
        
        assert false_positives == 0  # No false positives after trust is built