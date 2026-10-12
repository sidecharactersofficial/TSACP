use fast_loop::session::{FeatureVector, N_DIM};
use fast_loop::rotation::apply_rotation;
use fast_loop::params::Params;

#[test]
fn test_givens_rotation_known_angle() {
    let mut v_t = [0.0; N_DIM];
    v_t[0] = 1.0;
    
    let mut v_prev = [0.0; N_DIM];
    v_prev[0] = 1.0;
    v_prev[1] = 1.0;
    
    let x_t = FeatureVector { values: v_t };
    let x_prev = FeatureVector { values: v_prev };
    
    let params = Params::default();
    let delta_theta = std::f64::consts::PI / 2.0;
    
    let rotated = apply_rotation(&x_t, &x_t, Some(&x_prev), delta_theta, &params);
    
    assert!((rotated.values[0] - 0.0).abs() < 1e-9);
    assert!((rotated.values[1] - 1.0).abs() < 1e-9);
    for i in 2..N_DIM {
        assert!((rotated.values[i] - 0.0).abs() < 1e-9);
    }
}

#[test]
fn test_givens_rotation_preserves_orthogonal() {
    let mut v_t = [0.0; N_DIM];
    v_t[0] = 1.0;
    
    let mut v_prev = [0.0; N_DIM];
    v_prev[0] = 1.0;
    v_prev[1] = 1.0;
    
    let plane_v_t = FeatureVector { values: v_t };
    let plane_v_prev = FeatureVector { values: v_prev };
    
    // Create an orthogonal vector (in Z direction, since span is X-Y)
    let mut orthogonal = [0.0; N_DIM];
    orthogonal[2] = 5.0; // arbitrary magnitude
    let x_ortho = FeatureVector { values: orthogonal };
    
    let params = Params::default();
    let delta_theta = std::f64::consts::PI / 4.0;
    
    let rotated = apply_rotation(&x_ortho, &plane_v_t, Some(&plane_v_prev), delta_theta, &params);
    
    // It should be completely unchanged
    assert!((rotated.values[0] - 0.0).abs() < 1e-9);
    assert!((rotated.values[1] - 0.0).abs() < 1e-9);
    assert!((rotated.values[2] - 5.0).abs() < 1e-9);
    for i in 3..N_DIM {
        assert!((rotated.values[i] - 0.0).abs() < 1e-9);
    }
}
