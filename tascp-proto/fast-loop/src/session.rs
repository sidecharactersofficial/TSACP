use crate::params::Params;
use arc_swap::ArcSwap;
use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;
use chrono::{DateTime, Utc};
use std::time::Duration;

pub const N_DIM: usize = 10;
pub const WINDOW_SIZE_W: usize = 10;

#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub values: [f64; N_DIM],
}

pub struct SessionState {
    pub params: Arc<ArcSwap<Params>>,
    pub window: Mutex<VecDeque<(FeatureVector, FeatureVector, f64)>>, // (feature, velocity, m_t)
    pub last_accessed: Mutex<DateTime<Utc>>,
}

impl SessionState {
    pub fn new(params: Params) -> Self {
        Self {
            params: Arc::new(ArcSwap::from_pointee(params)),
            window: Mutex::new(VecDeque::with_capacity(WINDOW_SIZE_W)),
            last_accessed: Mutex::new(Utc::now()),
        }
    }
}

pub type SessionStore = Arc<DashMap<String, Arc<SessionState>>>;

/// Periodic task to sweep and evict stale sessions (e.g. inactive for N seconds)
pub async fn sweep_stale_sessions(store: SessionStore, max_idle_secs: i64, interval_secs: u64) {
    let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
    loop {
        interval.tick().await;
        let now = Utc::now();
        let mut to_remove = Vec::new();
        
        for entry in store.iter() {
            let state = entry.value();
            let last_accessed = *state.last_accessed.lock().await;
            if (now - last_accessed).num_seconds() > max_idle_secs {
                to_remove.push(entry.key().clone());
            }
        }
        
        for key in to_remove {
            store.remove(&key);
        }
    }
}
