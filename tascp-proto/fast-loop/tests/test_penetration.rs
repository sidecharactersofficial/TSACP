use fast_loop::session::{FeatureVector, N_DIM};
use fast_loop::penetration::compute_penetration;
use std::collections::VecDeque;

#[test]
fn test_penetration_converged_pgd() {
    let mut window = VecDeque::new();
    
    for _ in 0..5 {
        let mut v_i = [0.0; N_DIM];
        v_i[0] = 1.0; 
        window.push_back((
            FeatureVector { values: [0.0; N_DIM] },
            FeatureVector { values: v_i },
            1.0 // M_i
        ));
    }
    
    let epsilon_p = compute_penetration(&window);
    assert!((epsilon_p - 1.0).abs() < 1e-9, "Epsilon P should be high for converged PGD, got {}", epsilon_p);
}

#[test]
fn test_penetration_benign_traffic() {
    let mut window = VecDeque::new();
    
    for _ in 0..5 {
        let mut v_i = [0.0; N_DIM];
        v_i[0] = 1.0;
        window.push_back((
            FeatureVector { values: [0.0; N_DIM] },
            FeatureVector { values: v_i },
            0.0 // M_i
        ));
    }
    
    let epsilon_p = compute_penetration(&window);
    assert!((epsilon_p - 0.0).abs() < 1e-9, "Epsilon P should be low for benign traffic, got {}", epsilon_p);
}

#[test]
fn test_penetration_window_size_invariance() {
    let create_window = |size| {
        let mut window = VecDeque::new();
        for _ in 0..size {
            let mut v_i = [0.0; N_DIM];
            v_i[0] = 2.0; 
            window.push_back((
                FeatureVector { values: [0.0; N_DIM] },
                FeatureVector { values: v_i },
                0.5 // M_i
            ));
        }
        window
    };
    
    let ep_small = compute_penetration(&create_window(5));
    let ep_large = compute_penetration(&create_window(50));
    
    assert!((ep_small - ep_large).abs() < 1e-9, "Epsilon P should be invariant to window size");
    assert!((ep_small - 1.0).abs() < 1e-9); 
}
