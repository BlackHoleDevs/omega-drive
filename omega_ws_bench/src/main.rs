use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH, Duration};
use tokio::net::TcpStream;
use tokio::io::AsyncWriteExt;
use futures_util::{StreamExt, SinkExt};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "omega_ws_bench", author, version, about = "Omega Drive WebSocket Stress Testing Tool")]
struct Args {
    /// WebSocket Server URL
    #[arg(short, long, default_value = "ws://127.0.0.1:8082")]
    url: String,

    /// Redis/Omega TCP Port for Publishing updates
    #[arg(long, default_value = "127.0.0.1:6380")]
    redis_addr: String,

    /// Number of concurrent clients to connect
    #[arg(short, long, default_value_t = 1000)]
    connections: usize,

    /// Key/Topic to subscribe to
    #[arg(short, long, default_value = "stress_test_channel")]
    key: String,

    /// Delay in milliseconds between spawning connections to avoid TCP connection storms
    #[arg(long, default_value_t = 5)]
    stagger_ms: u64,

    /// Optional self-publish rate in messages per second. If 0, only listens.
    #[arg(short, long, default_value_t = 0)]
    publish_rate: usize,

    /// Message size in bytes when publishing
    #[arg(short = 's', long, default_value_t = 128)]
    message_size: usize,

    /// Test duration in seconds (0 for infinite)
    #[arg(short, long, default_value_t = 0)]
    duration: u64,
}

struct Stats {
    connected: AtomicU64,
    failed: AtomicU64,
    messages_received: AtomicU64,
    bytes_received: AtomicU64,
    total_latency_us: AtomicU64,
    latency_count: AtomicU64,
    max_latency_us: AtomicU64,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("⚡ OMEGA DRIVE WebSocket Stress Tester ⚡");
    println!("=========================================");
    println!("Target WS URL:       {}", args.url);
    println!("Target Redis/TCP:    {}", args.redis_addr);
    println!("Total Connections:   {}", args.connections);
    println!("Subscription Key:    {}", args.key);
    println!("Stagger Interval:    {} ms", args.stagger_ms);
    if args.publish_rate > 0 {
        println!("Self-Publishing:     {} msg/sec", args.publish_rate);
        println!("Publish Msg Size:    {} bytes", args.message_size.max(8));
    } else {
        println!("Self-Publishing:     Disabled (Listen-only mode)");
    }
    if args.duration > 0 {
        println!("Duration Limit:      {} seconds", args.duration);
    } else {
        println!("Duration Limit:      Infinite (Press Ctrl+C to stop)");
    }
    println!("=========================================\n");

    let stats = Arc::new(Stats {
        connected: AtomicU64::new(0),
        failed: AtomicU64::new(0),
        messages_received: AtomicU64::new(0),
        bytes_received: AtomicU64::new(0),
        total_latency_us: AtomicU64::new(0),
        latency_count: AtomicU64::new(0),
        max_latency_us: AtomicU64::new(0),
    });

    // 1. Spawn Publisher Task if enabled
    if args.publish_rate > 0 {
        let key = args.key.clone();
        let rate = args.publish_rate;
        let size = args.message_size.max(8);
        let redis_addr = args.redis_addr.clone();
        tokio::spawn(async move {
            run_publisher(&redis_addr, &key, rate, size).await;
        });
    }

    // 2. Spawn WS Connection Tasks
    let stagger = Duration::from_millis(args.stagger_ms);
    for _id in 0..args.connections {
        let url = args.url.clone();
        let key = args.key.clone();
        let stats_clone = Arc::clone(&stats);

        tokio::spawn(async move {
            loop {
                match connect_and_listen(&url, &key, Arc::clone(&stats_clone)).await {
                    Ok(_) => {
                        // Clean disconnect or shutdown
                        break;
                    }
                    Err(_) => {
                        stats_clone.failed.fetch_add(1, Ordering::Relaxed);
                        // Backoff and reconnect
                        tokio::time::sleep(Duration::from_secs(2)).await;
                    }
                }
            }
        });

        if stagger.as_nanos() > 0 {
            tokio::time::sleep(stagger).await;
        }
    }

    // 3. Spawn Statistics Reporter Task
    let stats_clone = Arc::clone(&stats);
    let start_time = Instant::now();
    let duration = args.duration;
    
    tokio::spawn(async move {
        let mut last_instant = Instant::now();
        let mut last_msg_count = 0;
        let mut last_bytes_count = 0;

        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            
            let now = Instant::now();
            let elapsed_sec = now.duration_since(last_instant).as_secs_f64();
            last_instant = now;

            let conn = stats_clone.connected.load(Ordering::Relaxed);
            let fail = stats_clone.failed.load(Ordering::Relaxed);
            let msg_total = stats_clone.messages_received.load(Ordering::Relaxed);
            let bytes_total = stats_clone.bytes_received.load(Ordering::Relaxed);

            let msg_diff = msg_total.saturating_sub(last_msg_count);
            let bytes_diff = bytes_total.saturating_sub(last_bytes_count);
            last_msg_count = msg_total;
            last_bytes_count = bytes_total;

            let msg_rate = msg_diff as f64 / elapsed_sec;
            let mb_rate = (bytes_diff as f64 / 1024.0 / 1024.0) * 8.0 / elapsed_sec; // Megabits/sec

            let lat_sum = stats_clone.total_latency_us.swap(0, Ordering::Relaxed);
            let lat_cnt = stats_clone.latency_count.swap(0, Ordering::Relaxed);
            let lat_max = stats_clone.max_latency_us.swap(0, Ordering::Relaxed);

            let avg_lat = if lat_cnt > 0 {
                format!("{:.2} ms", (lat_sum as f64 / lat_cnt as f64) / 1000.0)
            } else {
                "N/A".to_string()
            };

            let max_lat = if lat_max > 0 {
                format!("{:.2} ms", lat_max as f64 / 1000.0)
            } else {
                "N/A".to_string()
            };

            println!(
                "📈 Connections: {}/{} (Failed: {}) | Msg Rate: {:.1} msg/s ({:.2} Mbps) | Avg Latency: {} (Max: {})",
                conn, args.connections, fail, msg_rate, mb_rate, avg_lat, max_lat
            );

            if duration > 0 && start_time.elapsed().as_secs() >= duration {
                println!("\n⏱️ Test duration reached ({}s). Exiting.", duration);
                std::process::exit(0);
            }
        }
    });

    // Wait until duration completes (or wait forever)
    if duration > 0 {
        tokio::time::sleep(Duration::from_secs(duration)).await;
    } else {
        loop {
            tokio::time::sleep(Duration::from_secs(3600)).await;
        }
    }
}

async fn connect_and_listen(url: &str, key: &str, stats: Arc<Stats>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();

    // Subscribe to key
    let sub_cmd = serde_json::json!({
        "action": "subscribe",
        "key": key
    }).to_string();

    write.send(Message::Text(sub_cmd)).await?;

    // Wait for Ack
    if let Some(Ok(Message::Binary(bytes))) = read.next().await {
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if val.get("status").and_then(|s| s.as_str()) == Some("subscribed") {
                stats.connected.fetch_add(1, Ordering::Relaxed);
            } else {
                return Err("Failed to subscribe ack".into());
            }
        } else {
            return Err("Invalid JSON Ack".into());
        }
    } else {
        return Err("Connection closed before subscribe Ack".into());
    }

    // Main read loop
    while let Some(msg) = read.next().await {
        let msg = msg?;
        match msg {
            Message::Binary(bytes) => {
                stats.messages_received.fetch_add(1, Ordering::Relaxed);
                stats.bytes_received.fetch_add(bytes.len() as u64, Ordering::Relaxed);

                // Latency estimation
                if bytes.len() >= 8 {
                    let mut ts_bytes = [0u8; 8];
                    ts_bytes.copy_from_slice(&bytes[0..8]);
                    let sent_us = u64::from_be_bytes(ts_bytes);
                    let now_us = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_micros() as u64;

                    if now_us >= sent_us {
                        let latency = now_us - sent_us;
                        stats.total_latency_us.fetch_add(latency, Ordering::Relaxed);
                        stats.latency_count.fetch_add(1, Ordering::Relaxed);

                        // Running max
                        let mut current_max = stats.max_latency_us.load(Ordering::Relaxed);
                        while latency > current_max {
                            match stats.max_latency_us.compare_exchange_weak(
                                current_max,
                                latency,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(actual) => current_max = actual,
                            }
                        }
                    }
                }
            }
            Message::Close(_) => {
                break;
            }
            _ => {}
        }
    }

    stats.connected.fetch_sub(1, Ordering::Relaxed);
    Ok(())
}

async fn run_publisher(redis_addr: &str, key: &str, rate: usize, size: usize) {
    println!("📢 Publisher task online. Connecting to Omega Drive TCP on {}...", redis_addr);
    let mut stream = match TcpStream::connect(redis_addr).await {
        Ok(s) => s,
        Err(e) => {
            println!("❌ Publisher failed to connect to Omega Drive: {}. Updates will not be published.", e);
            return;
        }
    };
    println!("✅ Publisher connected! Spawning updates at {} msg/s.", rate);

    let interval = Duration::from_nanos((1_000_000_000 / rate) as u64);
    let mut ticker = tokio::time::interval(interval);
    
    // We reuse this buffer to build RESP SET command fast
    let mut payload = vec![0u8; size];

    loop {
        ticker.tick().await;

        // Embed timestamp in first 8 bytes (Big Endian)
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        payload[0..8].copy_from_slice(&now_us.to_be_bytes());

        // Construct RESP SET command
        // *3\r\n$3\r\nSET\r\n$<k_len>\r\n<key>\r\n$<p_len>\r\n<payload>\r\n
        let mut cmd = Vec::new();
        cmd.extend_from_slice(b"*3\r\n$3\r\nSET\r\n");
        cmd.extend_from_slice(format!("${}\r\n{}\r\n", key.len(), key).as_bytes());
        cmd.extend_from_slice(format!("${}\r\n", size).as_bytes());
        cmd.extend_from_slice(&payload);
        cmd.extend_from_slice(b"\r\n");

        if let Err(e) = stream.write_all(&cmd).await {
            println!("❌ Publisher write error: {}. Reconnecting...", e);
            // Quick reconnect attempt
            tokio::time::sleep(Duration::from_secs(1)).await;
            if let Ok(s) = TcpStream::connect(redis_addr).await {
                stream = s;
                println!("✅ Publisher reconnected successfully.");
            }
        }
    }
}
