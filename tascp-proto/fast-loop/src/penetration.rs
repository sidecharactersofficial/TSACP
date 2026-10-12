use crate::session::{FeatureVector, N_DIM};
use std::collections::VecDeque;

/// Computes epsilon_p (penetration) using windowed-average formula.
/// Uses Eq. "penetration" (Section IV-B) from the TASCP paper:
/// epsilon_p = (1 / |W|) * sum_{i in W} ||v_i||_2 * max(0, M_i)
pub fn compute_penetration(window: &VecDeque<(FeatureVector, FeatureVector, f64)>) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    
    let mut sum = 0.0;
    for (_, v_i, m_i) in window.iter() {
        let mut dist_sq = 0.0;
        for j in 0..N_DIM {
            dist_sq += v_i.values[j] * v_i.values[j];
        }
        sum += dist_sq.sqrt() * m_i.max(0.0);
    }
    
    sum / (window.len() as f64)
}
