use crate::session::{FeatureVector, N_DIM};
use crate::params::Params;
use nalgebra::SVector;

/// Applies the rotation R(delta_theta) to a given vector.
/// The rotation plane is defined by span{plane_v_t, plane_v_prev}.
/// Uses actual Givens rotation (Lemma 2 / Eq. 8 from the TASCP paper).
pub fn apply_rotation(
    vector_to_rotate: &FeatureVector, 
    plane_v_t: &FeatureVector, 
    plane_v_prev: Option<&FeatureVector>, 
    delta_theta: f64, 
    _params: &Params
) -> FeatureVector {
    if delta_theta.abs() < 1e-9 || plane_v_prev.is_none() {
        return vector_to_rotate.clone();
    }
    let prev = plane_v_prev.unwrap();

    let p_t = SVector::<f64, N_DIM>::from_row_slice(&plane_v_t.values);
    let p_prev = SVector::<f64, N_DIM>::from_row_slice(&prev.values);
    let target = SVector::<f64, N_DIM>::from_row_slice(&vector_to_rotate.values);

    let norm_t = p_t.norm();
    if norm_t < 1e-9 {
        return vector_to_rotate.clone();
    }

    let u = p_t / norm_t;
    // p_prev projected onto orthogonal complement of u
    let proj = p_prev.dot(&u) * u;
    let w = p_prev - proj;
    let norm_w = w.norm();

    if norm_w < 1e-9 {
        return vector_to_rotate.clone(); // Vectors are collinear, no unique rotation plane
    }

    let v_t_perp = w / norm_w;

    // Component of target in the plane
    let comp_u = target.dot(&u);
    let comp_perp = target.dot(&v_t_perp);
    
    // The component orthogonal to the plane remains unchanged
    let orthogonal_component = target - (comp_u * u) - (comp_perp * v_t_perp);

    // Rotate the in-plane components
    let rotated_u = comp_u * delta_theta.cos() - comp_perp * delta_theta.sin();
    let rotated_perp = comp_u * delta_theta.sin() + comp_perp * delta_theta.cos();

    let rotated = orthogonal_component + rotated_u * u + rotated_perp * v_t_perp;

    let mut out = [0.0; N_DIM];
    out.copy_from_slice(rotated.as_slice());
    FeatureVector { values: out }
}
