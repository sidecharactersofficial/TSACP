use std::sync::Arc;
use tokio::time::Duration;
use dashmap::DashMap;
use tokio::sync::Barrier;

use fast_loop::params::Params;
use fast_loop::session::{SessionState, SessionStore, FeatureVector, N_DIM};
use fast_loop::defense::process_request;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_update_does_not_block() {
    let sessions: SessionStore = Arc::new(DashMap::new());
    let session_id = "test_session".to_string();
    let session = Arc::new(SessionState::new(Params::default()));
    sessions.insert(session_id.clone(), session.clone());

    let num_tasks = 10;
    // Barrier to synchronize all 10 worker tasks + 1 param update task + 1 main task
    let barrier = Arc::new(Barrier::new(num_tasks + 2));
    let mut join_handles = Vec::new();

    // Spawn concurrent tasks simulating high-frequency traffic hitting the fast path
    for i in 0..num_tasks {
        let session_clone = session.clone();
        let b = barrier.clone();
        join_handles.push(tokio::spawn(async move {
            b.wait().await; // Synchronize start
            for j in 0..200 {
                let mut features = [0.0; N_DIM];
                features[0] = (i * 100 + j) as f64;
                let x_t = FeatureVector { values: features };
                
                // This should never block waiting for a parameter update
                let _ = process_request(session_clone.clone(), x_t).await;
            }
        }));
    }

    // Task to simulate slow loop pushing an update exactly while tasks are running
    let session_clone2 = session.clone();
    let b_update = barrier.clone();
    let update_task = tokio::spawn(async move {
        b_update.wait().await; // Synchronize start
        
        // Wait a tiny fraction of time to ensure we interleave mid-flight with the requests
        tokio::time::sleep(Duration::from_millis(2)).await; 
        
        let new_params = Params {
            gamma: 0.1,
            lambda: 0.2,
            k: 3.0,
            tau: 0.4,
        };
        // The admin update uses arc_swap for a lock-free pointer swap
        session_clone2.params.store(Arc::new(new_params));
    });

    barrier.wait().await; // Release the hounds

    for handle in join_handles {
        let _ = handle.await;
    }
    let _ = update_task.await;

    // Verify params were actually updated
    let final_params = session.params.load();
    assert_eq!(final_params.gamma, 0.1);
}
