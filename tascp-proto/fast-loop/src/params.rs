use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Params {
    pub gamma: f64,
    pub lambda: f64,
    pub k: f64,
    pub tau: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            gamma: 0.9,
            lambda: 0.1,
            k: 5.0,
            tau: 0.5,
        }
    }
}
