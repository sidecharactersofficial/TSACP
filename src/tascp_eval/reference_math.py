# Python re-implementation of momentum/penetration/rotation.
# EXPLICITLY matching the corrected windowed epsilon_p formula and single-step momentum.
import numpy as np

def compute_momentum(vel_t, vel_prev):
    '''Single step M_t (Eq 4-5)'''
    norm_t = np.linalg.norm(vel_t)
    norm_prev = np.linalg.norm(vel_prev)
    if norm_t > 1e-9 and norm_prev > 1e-9:
        return np.dot(vel_t, vel_prev) / (norm_t * norm_prev)
    return 0.0

def compute_penetration(window):
    '''
    window: list of tuples (v_i, m_i)
    epsilon_p = (1 / |W|) * sum_{i in W} ||v_i||_2 * max(0, M_i)
    '''
    if not window:
        return 0.0
    
    total = 0.0
    for v_i, m_i in window:
        total += np.linalg.norm(v_i) * max(0.0, m_i)
    return total / len(window)

# TODO: implement apply_rotation reference logic here matching rust's apply_rotation.
