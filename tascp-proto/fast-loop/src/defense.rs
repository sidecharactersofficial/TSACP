use crate::session::{SessionState, FeatureVector, N_DIM};
use crate::momentum::compute_momentum;
use crate::penetration::compute_penetration;
use crate::rotation::apply_rotation;
use chrono::Utc;
use std::sync::Arc;

pub async fn process_request(state: Arc<SessionState>, x_t: FeatureVector) -> (FeatureVector, f64, f64, f64) {
    let params = state.params.load().as_ref().clone();
    
    let mut window = state.window.lock().await;
    
    // Update last accessed
    *state.last_accessed.lock().await = Utc::now();
    
    let x_prev = window.back().map(|(prev_x, _, _)| prev_x.clone());
    let v_prev = window.back().map(|(_, prev_v, _)| prev_v.clone());
    
    // Compute velocity (x_t - x_{t-1})
    let velocity = if let Some(last) = &x_prev {
        let mut v = [0.0; N_DIM];
        for i in 0..N_DIM {
            v[i] = x_t.values[i] - last.values[i];
        }
        FeatureVector { values: v }
    } else {
        FeatureVector { values: [0.0; N_DIM] }
    };
    
    let m_t = if let Some(vp) = v_prev {
        compute_momentum(&velocity, &vp)
    } else {
        0.0
    };
    
    window.push_back((x_t.clone(), velocity.clone(), m_t));
    if window.len() > crate::session::WINDOW_SIZE_W {
        window.pop_front();
    }
    
    let epsilon_p = compute_penetration(&window);
    
    // Archimedean / Logarithmic phase logic (Eq. Penetration Epsilon, Sec IV-B revision)
    let delta_theta = if epsilon_p <= params.gamma {
        params.lambda * epsilon_p
    } else {
        params.lambda * (params.k * (epsilon_p - params.gamma)).exp()
    };
    
    let x_defended = apply_rotation(&x_t, &x_t, x_prev.as_ref(), delta_theta, &params);
    
    (x_defended, m_t, epsilon_p, delta_theta)
}
