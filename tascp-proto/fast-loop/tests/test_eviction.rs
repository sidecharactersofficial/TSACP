use std::sync::Arc;
use dashmap::DashMap;
use chrono::{Utc, TimeDelta};
use tokio::time::Duration;

use fast_loop::params::Params;
use fast_loop::session::{SessionState, SessionStore, sweep_stale_sessions};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_session_eviction_and_race() {
    let sessions: SessionStore = Arc::new(DashMap::new());
    
    // Create an old session and a new session
    let old_session = Arc::new(SessionState::new(Params::default()));
    // Artificially age the old session
    *old_session.last_accessed.lock().await = Utc::now() - TimeDelta::seconds(100);
    sessions.insert("old_session".to_string(), old_session.clone());

    let new_session = Arc::new(SessionState::new(Params::default()));
    sessions.insert("new_session".to_string(), new_session.clone());

    // Start sweep task with 60s max idle, interval of 1 second for test purposes
    let sessions_clone = sessions.clone();
    let sweep_task = tokio::spawn(async move {
        sweep_stale_sessions(sessions_clone, 60, 1).await;
    });

    // Wait slightly more than 1 second to allow the sweep task to run at least once
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // The old session should be evicted
    assert!(sessions.get("old_session").is_none(), "Old session was not evicted");
    
    // The new session should remain
    assert!(sessions.get("new_session").is_some(), "New session was incorrectly evicted");

    // Abort the sweep task since it runs indefinitely
    sweep_task.abort();
}
