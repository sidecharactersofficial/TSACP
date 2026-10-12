use axum::{
    routing::post,
    Router, Json, extract::{State, Path},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpListener;
use chrono::Utc;

use fast_loop::params::Params;
use fast_loop::session::{SessionState, SessionStore, sweep_stale_sessions, FeatureVector, N_DIM};
use fast_loop::event_store::{EventStore, Event};
use fast_loop::defense::process_request;

#[derive(Clone)]
struct AppState {
    sessions: SessionStore,
    event_store: Arc<EventStore>,
}

#[derive(Deserialize)]
struct PredictRequest {
    session_id: String,
    features: [f64; N_DIM],
}

#[derive(Serialize)]
struct PredictResponse {
    defended_features: [f64; N_DIM],
}

async fn handle_predict(State(state): State<AppState>, Json(payload): Json<PredictRequest>) -> Json<PredictResponse> {
    let session_id = payload.session_id.clone();
    
    let session = state.sessions
        .entry(session_id.clone())
        .or_insert_with(|| Arc::new(SessionState::new(Params::default())))
        .clone();
        
    let x_t = FeatureVector { values: payload.features };
    
    let (x_defended, m_t, epsilon_p, delta_theta) = process_request(session, x_t).await;
    
    state.event_store.push(Event {
        session_id,
        timestamp_ms: Utc::now().timestamp_millis(),
        feature_vector: payload.features,
        m_t,
        epsilon_p,
        delta_theta,
        classification_label: None,
    });
    
    Json(PredictResponse {
        defended_features: x_defended.values,
    })
}

async fn handle_admin_params(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(params): Json<Params>
) -> &'static str {
    if let Some(session) = state.sessions.get(&session_id) {
        session.params.store(Arc::new(params));
        "Updated"
    } else {
        "Session not found"
    }
}

#[tokio::main]
async fn main() {
    let sessions: SessionStore = Arc::new(dashmap::DashMap::new());
    let event_store = Arc::new(EventStore::new("events.db".to_string()));
    
    let app_state = AppState {
        sessions: sessions.clone(),
        event_store,
    };
    
    tokio::spawn(sweep_stale_sessions(sessions.clone(), 60, 10));
    
    let app = Router::new()
        .route("/predict", post(handle_predict))
        .route("/admin/params/{session_id}", post(handle_admin_params))
        .with_state(app_state);
        
    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Fast loop listening on 0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
}
