use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::broadcast;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;
use futures_util::{StreamExt, SinkExt};
use serde::Deserialize;

#[derive(Deserialize)]
struct WsAction {
    action: String,
    key: String,
}

type TopicRegistry = Arc<DashMap<String, broadcast::Sender<Vec<u8>>>>;

lazy_static::lazy_static! {
    // Global distribution channels registry (Pub/Sub)
    static ref TOPIC_REGISTRY: TopicRegistry = Arc::new(DashMap::new());
}

// Function capturing data from the main OmegaDrive database
pub fn broadcast_update(key: &str, payload: &[u8]) {
    // If anyone is listening to the given key, immediately broadcast the payload!
    if let Some(sender) = TOPIC_REGISTRY.get(key) {
        let _ = sender.send(payload.to_vec()); // Instant non-blocking broadcast
    }
}

pub fn start_websocket_server(port: u16, num_workers: usize) {
    std::thread::spawn(move || {
        // Enforcing Neural License Limits on the WebSocket background process!
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_workers.max(1)) // At least 1 thread needed, but bounded by license
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async move {
            let addr = format!("0.0.0.0:{}", port);
            let listener = TcpListener::bind(&addr).await.expect("Failed to bind WebSocket listener");
            println!("🚀 [ACCELERATOR] WebSocket Pub/Sub Server active on ws://{} [License Bounds: {} Cores]", addr, num_workers);

            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(handle_connection(stream));
            }
        });
    });
}

async fn handle_connection(raw_stream: TcpStream) {
    // Handshake phase with a new browser client
    let ws_stream = match accept_async(raw_stream).await {
        Ok(ws) => ws,
        Err(_) => return, // Client rejected / bad protocol
    };

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    
    // MPSC used to consolidate traffic from different channels (keys) into 1 WebSocket socket
    let (client_tx, mut client_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(1024);
    
    // Pusher Thread: Sends messages from MPSC directly to the client's Socket
    tokio::spawn(async move {
        while let Some(msg) = client_rx.recv().await {
            // If the client disconnects, send() will return Err and the sending process will close silently in the background (Zero memory leak)
            if ws_sender.send(Message::Binary(msg)).await.is_err() {
                break;
            }
        }
    });

    // Remember what the client is already subscribing to within 1 connection
    let mut active_subs = std::collections::HashSet::new();

    // Loop reading JSON commands from the WebSocket (Industry Standard)
    while let Some(msg) = ws_receiver.next().await {
        if let Ok(Message::Text(text)) = msg {
            if let Ok(req) = serde_json::from_str::<WsAction>(&text) {
                if req.action == "subscribe" && !active_subs.contains(&req.key) {
                    active_subs.insert(req.key.clone());
                    
                    // Find channel for key or create a new one (buffer for 10k messages per channel!)
                    let tx = TOPIC_REGISTRY.entry(req.key.clone()).or_insert_with(|| {
                        let (tx, _) = broadcast::channel(10000);
                        tx
                    }).value().clone();
                    
                    let mut rx = tx.subscribe();
                    let client_tx_clone = client_tx.clone();
                    
                    // Independent Subscriber Thread: Listens to global messages from Omega and pushes them into the client's pipeline
                    tokio::spawn(async move {
                        loop {
                            match rx.recv().await {
                                Ok(msg) => {
                                    if client_tx_clone.send(msg).await.is_err() {
                                        break; // Client disconnected, killing thread
                                    }
                                }
                                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                                    // Web browser cannot keep up reading at Omega's speed!
                                    // Loses packets, but we CANNOT close the connection - continue.
                                    continue;
                                }
                                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                                    break; // Channel destroyed
                                }
                            }
                        }
                    });
                    
                    // Subscription confirmation for frontend
                    let ack = format!("{{\"status\":\"subscribed\",\"key\":\"{}\"}}", req.key);
                    let _ = client_tx.send(ack.into_bytes()).await;
                }
            }
        }
    }
}
