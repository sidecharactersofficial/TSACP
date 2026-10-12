use crate::session::{FeatureVector, N_DIM};

/// Computes the single-step momentum vector M_t (cosine similarity).
/// Uses Eq. 4-5 from the TASCP paper.
pub fn compute_momentum(vel_t: &FeatureVector, vel_prev: &FeatureVector) -> f64 {
    let mut dot = 0.0;
    let mut norm_t = 0.0;
    let mut norm_prev = 0.0;
    
    for j in 0..N_DIM {
        dot += vel_t.values[j] * vel_prev.values[j];
        norm_t += vel_t.values[j] * vel_t.values[j];
        norm_prev += vel_prev.values[j] * vel_prev.values[j];
    }
    
    norm_t = norm_t.sqrt();
    norm_prev = norm_prev.sqrt();
    
    if norm_t > 1e-9 && norm_prev > 1e-9 {
        dot / (norm_t * norm_prev)
    } else {
        0.0
    }
}
