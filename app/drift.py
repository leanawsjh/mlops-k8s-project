import numpy as np

def detect_drift(features):
    mean_val = np.mean(features)
    
    # simple rule-based drift detection
    if mean_val > 10:
        return True
    return False
