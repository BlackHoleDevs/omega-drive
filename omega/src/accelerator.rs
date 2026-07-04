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
    // Globalny rejestr kanałów dystrybucyjnych (Pub/Sub)
    static ref TOPIC_REGISTRY: TopicRegistry = Arc::new(DashMap::new());
}

// Funkcja przechwytująca dane z głównej bazy OmegaDrive
pub fn broadcast_update(key: &str, payload: &[u8]) {
    // Jeśli ktokolwiek słucha danego klucza, natychmiast wrzucamy payload w eter!
    if let Some(sender) = TOPIC_REGISTRY.get(key) {
        let _ = sender.send(payload.to_vec()); // Błyskawiczny non-blocking broadcast
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
    // Faza Handshake z nowym klientem przeglądarkowym
    let ws_stream = match accept_async(raw_stream).await {
        Ok(ws) => ws,
        Err(_) => return, // Klient odrzucony / zły protokół
    };

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    
    // MPSC używane żeby skondensować ruch z różnych kanałów (keys) do 1 gniazda WebSocket
    let (client_tx, mut client_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(1024);
    
    // Wątek-Niszczyciel (Pusher): Przesyła wiadomości z MPSC bezpośrednio do Socketu klienta
    tokio::spawn(async move {
        while let Some(msg) = client_rx.recv().await {
            // Jeśli klient się rozłączy, send() zwróci Err i proces wysyłający zamknie się cicho w tle (Zero memory leak)
            if ws_sender.send(Message::Binary(msg)).await.is_err() {
                break;
            }
        }
    });

    // Pamiętamy co klient już subskrybuje w ramach 1 połączenia
    let mut active_subs = std::collections::HashSet::new();

    // Pętla odczytu komend JSON z WebSocketu (Industry Standard)
    while let Some(msg) = ws_receiver.next().await {
        if let Ok(Message::Text(text)) = msg {
            if let Ok(req) = serde_json::from_str::<WsAction>(&text) {
                if req.action == "subscribe" && !active_subs.contains(&req.key) {
                    active_subs.insert(req.key.clone());
                    
                    // Znajdź kanał dla klucza lub stwórz nowy (bufor na 10k wiadomości na kanał!)
                    let tx = TOPIC_REGISTRY.entry(req.key.clone()).or_insert_with(|| {
                        let (tx, _) = broadcast::channel(10000);
                        tx
                    }).value().clone();
                    
                    let mut rx = tx.subscribe();
                    let client_tx_clone = client_tx.clone();
                    
                    // Niezależny Wątek Subskrybenta: Nasłuchuje na globalne krzyki z Omegi i wrzuca do rury klienta
                    tokio::spawn(async move {
                        loop {
                            match rx.recv().await {
                                Ok(msg) => {
                                    if client_tx_clone.send(msg).await.is_err() {
                                        break; // Klient uciekł, zabijamy wątek
                                    }
                                }
                                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                                    // Przeglądarka internetowa nie nadąża czytać z prędkością Omegi!
                                    // Gubi pakiety, ale NIE możemy zamykać połączenia - kontynuujemy.
                                    continue;
                                }
                                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                                    break; // Kanał zniszczony
                                }
                            }
                        }
                    });
                    
                    // Potwierdzenie podłączenia dla Frontendowca
                    let ack = format!("{{\"status\":\"subscribed\",\"key\":\"{}\"}}", req.key);
                    let _ = client_tx.send(ack.into_bytes()).await;
                }
            }
        }
    }
}
