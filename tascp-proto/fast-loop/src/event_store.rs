use crate::session::N_DIM;
use crossbeam_channel::{unbounded, Sender};
use rusqlite::Connection;
use std::thread;

pub struct Event {
    pub session_id: String,
    pub timestamp_ms: i64,
    pub feature_vector: [f64; N_DIM],
    pub m_t: f64,
    pub epsilon_p: f64,
    pub delta_theta: f64,
    pub classification_label: Option<String>,
}

pub struct EventStore {
    sender: Sender<Event>,
}

impl EventStore {
    pub fn new(db_path: String) -> Self {
        let (sender, receiver) = unbounded::<Event>();
        
        thread::spawn(move || {
            let mut conn = Connection::open(&db_path).expect("Failed to open DB");
            
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    feature_vector TEXT NOT NULL,
                    m_t REAL NOT NULL,
                    epsilon_p REAL NOT NULL,
                    delta_theta REAL NOT NULL,
                    classification_label TEXT
                )",
                (),
            ).expect("Failed to create schema");
            
            while let Ok(event) = receiver.recv() {
                let fv_json = serde_json::to_string(&event.feature_vector).unwrap();
                
                let res = conn.execute(
                    "INSERT INTO events (session_id, timestamp_ms, feature_vector, m_t, epsilon_p, delta_theta, classification_label) 
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    rusqlite::params![event.session_id, event.timestamp_ms, fv_json, event.m_t, event.epsilon_p, event.delta_theta, event.classification_label],
                );
                
                if let Err(e) = res {
                    eprintln!("Failed to insert event: {}", e);
                }
            }
        });
        
        Self { sender }
    }
    
    pub fn push(&self, event: Event) {
        let _ = self.sender.send(event);
    }
}
