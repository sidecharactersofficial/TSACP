//! In-process microbenchmark of the TASCP fast-loop hot path (P3-18).
//!
//! Motivation: the paper's 7.08 ms per-query latency was measured over HTTP
//! (dockerized prototype). The network hop and JSON serialization dominate
//! that figure, so it overstates the defense's compute cost. This benchmark
//! times ONLY the in-process defense computation — velocity, momentum,
//! penetration-integral, and the Givens rotation — removing the HTTP-overhead
//! caveat entirely.
//!
//! Run: cargo bench (from tascp-proto/fast-loop/)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fast_loop::params::Params;
use fast_loop::rotation::apply_rotation;
use fast_loop::momentum::compute_momentum;
use fast_loop::penetration::compute_penetration;
use fast_loop::session::{FeatureVector, WINDOW_SIZE_W};

use std::collections::VecDeque;

fn make_vec(seed: usize, scale: f64) -> FeatureVector {
    // deterministic pseudo-random fill (xorshift-style) — no external rng dep
    let mut s: u64 = 0x9E3779B97F4A7C15 ^ (seed as u64).wrapping_mul(0x2545F4914F6CDD1D);
    let mut values = [0.0; fast_loop::session::N_DIM];
    for v in values.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = ((s % 10_000) as f64 / 10_000.0 - 0.5) * scale;
    }
    FeatureVector { values }
}

fn bench_full_hot_path(c: &mut Criterion) {
    let params = Params::default();
    let mut window: VecDeque<(FeatureVector, FeatureVector, f64)> =
        VecDeque::with_capacity(WINDOW_SIZE_W);
    // pre-fill the window like a live session (10 entries)
    let mut prev_x = make_vec(0, 1.0);
    let mut prev_v = FeatureVector { values: [0.0; fast_loop::session::N_DIM] };
    for i in 0..WINDOW_SIZE_W {
        let x = make_vec(i + 1, 1.0);
        let mut v = FeatureVector { values: [0.0; fast_loop::session::N_DIM] };
        for j in 0..fast_loop::session::N_DIM {
            v.values[j] = x.values[j] - prev_x.values[j];
        }
        let m = compute_momentum(&v, &prev_v);
        window.push_back((x.clone(), v.clone(), m));
        if window.len() > WINDOW_SIZE_W {
            window.pop_front();
        }
        prev_x = x;
        prev_v = v;
    }

    c.bench_function("hot_path_single_query", |b| {
        b.iter(|| {
            let x_t = make_vec(999, 1.0);
            // --- exact fast-loop defense.rs sequence ---
            let x_prev = window.back().map(|(px, _, _)| px.clone());
            let v_prev = window.back().map(|(_, pv, _)| pv.clone());

            let mut velocity = FeatureVector { values: [0.0; fast_loop::session::N_DIM] };
            if let Some(last) = &x_prev {
                for i in 0..fast_loop::session::N_DIM {
                    velocity.values[i] = x_t.values[i] - last.values[i];
                }
            }
            let m_t = match &v_prev {
                Some(vp) => compute_momentum(black_box(&velocity), vp),
                None => 0.0,
            };
            window.push_back((x_t.clone(), velocity.clone(), m_t));
            if window.len() > WINDOW_SIZE_W {
                window.pop_front();
            }
            let epsilon_p = compute_penetration(&window);
            let delta_theta = if epsilon_p <= params.gamma {
                params.lambda * epsilon_p
            } else {
                params.lambda * (params.k * (epsilon_p - params.gamma)).exp()
            };
            let x_defended =
                apply_rotation(black_box(&x_t), black_box(&x_t), x_prev.as_ref(), delta_theta, &params);
            black_box(x_defended)
        })
    });
}

fn bench_components(c: &mut Criterion) {
    let params = Params::default();
    let v1 = make_vec(1, 1.0);
    let v2 = make_vec(2, 1.0);
    let x = make_vec(3, 1.0);

    c.bench_function("momentum", |b| b.iter(|| compute_momentum(black_box(&v1), black_box(&v2))));
    c.bench_function("rotation", |b| {
        b.iter(|| apply_rotation(black_box(&x), black_box(&x), Some(&v2), 0.3, &params))
    });
}

criterion_group!(benches, bench_full_hot_path, bench_components);
criterion_main!(benches);
