import numpy as np
from src.tascp_eval.reference_math import compute_penetration

def test_penetration_converged_pgd():
    # Matches Rust test: 5 items, ||v_i||=1.0, M_i=1.0
    v_i = np.zeros(10)
    v_i[0] = 1.0
    window = [(v_i, 1.0) for _ in range(5)]
    assert abs(compute_penetration(window) - 1.0) < 1e-9

def test_penetration_benign_traffic():
    # Matches Rust test: 5 items, ||v_i||=1.0, M_i=0.0
    v_i = np.zeros(10)
    v_i[0] = 1.0
    window = [(v_i, 0.0) for _ in range(5)]
    assert abs(compute_penetration(window) - 0.0) < 1e-9

def test_penetration_window_size_invariance():
    # Matches Rust test: ||v_i||=2.0, M_i=0.5, sizes 5 vs 50
    v_i = np.zeros(10)
    v_i[0] = 2.0
    
    def create_window(size):
        return [(v_i, 0.5) for _ in range(size)]
        
    ep_small = compute_penetration(create_window(5))
    ep_large = compute_penetration(create_window(50))
    
    assert abs(ep_small - ep_large) < 1e-9
    assert abs(ep_small - 1.0) < 1e-9
