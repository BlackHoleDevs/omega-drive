mod inference;
mod resp;
mod telemetry;
mod accelerator;
mod hash_engine;
mod vector_store;
mod ws_opt;
use dashmap::DashMap;
use inference::{McnnModel, Device};
use resp::{Command, parse_command_fast, make_pong, make_int, make_bulk_string};
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::UnixListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::sync::Arc;
use bytes::{BytesMut, Buf};
use clap::Parser;

use socket2::{Socket, Domain, Type, Protocol, SockAddr};
use std::net::SocketAddr;
use std::fs;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "cpu")]
    device: String,
    #[arg(short, long, default_value_t = 6380)]
    port: u16,
    #[arg(short, long)]
    workers: Option<usize>,
    #[arg(short = 's', long)]
    statistics: bool,
    #[arg(long)]
    hdd: Option<u64>,
    #[arg(long, default_value_t = 8082)]
    ws_port: u16,
    
    // Redis-compatible daemonization and socket args
    #[arg(long, default_value = "127.0.0.1")]
    bind: String,
    #[arg(long)]
    unixsocket: Option<String>,
    #[arg(long)]
    unixsocketperm: Option<String>,
    #[arg(long, default_value = "no")]
    daemonize: String,
}

#[derive(Debug, Eq, Clone, serde::Serialize, serde::Deserialize)]
pub enum AirKey {
    Inline([u8; 15], u8),
    Boxed(Box<[u8]>),
}

impl AirKey {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.len() <= 15 {
            let mut data = [0u8; 15];
            data[..bytes.len()].copy_from_slice(bytes);
            AirKey::Inline(data, bytes.len() as u8)
        } else {
            AirKey::Boxed(Box::from(bytes))
        }
    }
    pub(crate) fn as_bytes(&self) -> &[u8] {
        match self {
            AirKey::Inline(data, len) => &data[..*len as usize],
            AirKey::Boxed(b) => b,
        }
    }
}

impl std::hash::Hash for AirKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.as_bytes().hash(state); }
}

impl PartialEq for AirKey {
    fn eq(&self, other: &Self) -> bool { self.as_bytes() == other.as_bytes() }
}

use serde_big_array::BigArray;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
struct ChunkData(#[serde(with = "BigArray")] pub [u8; 64]);

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum NeuralValue {
    Single { best_idx: u16, data: ChunkData, len: usize },
    Multi { original_len: usize, best_idx: u16, chunks: Vec<ChunkData> },
    FlatHash(Vec<u8>),
}


const RESP_OK: &[u8] = b"+OK\r\n";
const RESP_CRLF: &[u8] = b"\r\n";
const RESP_NULL: &[u8] = b"$-1\r\n";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    let device = match args.device.as_str() {
        "gpu" => Device::Gpu,
        "hybrid" => Device::Hybrid,
        _ => Device::Cpu,
    };

    let weights_paths = ["logic_gate.omm", "airdb_core/logic_gate.omm", "../logic_gate.omm"];
    let mut model_path = weights_paths[0];
    for path in &weights_paths { if std::path::Path::new(path).exists() { model_path = path; break; } }
    
    let model_base = match McnnModel::load(model_path, device) {
        Ok(m) => Arc::new(m),
        Err(e) => {
            println!("❌ FATAL: Neural Integrity Check Failed! {}", e);
            std::process::exit(1);
        }
    };

    let total_cores = num_cpus::get();
    let limit = model_base.worker_limit;
    let num_workers = args.workers.unwrap_or(total_cores).min(limit).min(total_cores);

    println!("🚀 OMEGA DRIVE 3.0 - HYBRID NEURAL GATEWAY");
    if limit == 2 {
        println!("🧬 Neural License Verified | Active: {}/{} Workers [LITE TIER - 2 Cores Bound]", num_workers, total_cores);
    } else if limit == 4 {
        println!("🧬 Neural License Verified | Active: {}/{} Workers [PAY-AS-YOU-WANT TIER - 4 Cores Bound]", num_workers, total_cores);
    } else if limit == 8 {
        println!("🧬 Neural License Verified | Active: {}/{} Workers [PAY-AS-YOU-WANT TIER - 8 Cores Bound]", num_workers, total_cores);
    } else {
        println!("🧬 Neural License Verified | Active: {}/{} Workers [UNLIMITED PERFORMANCE TIER]", num_workers, total_cores);
    }

    #[cfg(unix)]
    if args.daemonize == "yes" {
        println!("🚀 Daemonizing Omega Drive...");
        unsafe {
            if libc::daemon(0, 0) != 0 {
                eprintln!("❌ Failed to daemonize process");
                std::process::exit(1);
            }
        }
    }

    #[cfg(unix)]
    let uds_listener_base = {
        let socket_path = args.unixsocket.as_deref().unwrap_or("/tmp/omega.sock");
        let _ = fs::remove_file(socket_path);
        let base = Arc::new(std::os::unix::net::UnixListener::bind(socket_path).expect("failed to bind uds"));
        base.set_nonblocking(true).expect("failed to set nonblocking on uds");
        
        if let Some(perm_str) = &args.unixsocketperm {
            use std::os::unix::fs::PermissionsExt;
            let perm_val = u32::from_str_radix(perm_str, 8).unwrap_or(0o777);
            let _ = fs::set_permissions(socket_path, fs::Permissions::from_mode(perm_val));
        }
        
        base
    };

    let mut threads = Vec::new();
    let global_db = Arc::new(DashMap::with_capacity_and_hasher_and_shard_amount(1_000_000, fxhash::FxBuildHasher::default(), 1024));
    let vector_db = Arc::new(crate::vector_store::SharedVectorStore::new(1536));

    let dump_file = "omegadrive.dump";
    if std::path::Path::new(dump_file).exists() {
        println!("💾 Loading persistent database from {}...", dump_file);
        if let Ok(data) = fs::read(dump_file) {
            if let Ok(loaded) = bincode::deserialize::<Vec<(AirKey, NeuralValue)>>(&data) {
                for (k, v) in loaded { global_db.insert(k, v); }
                println!("✅ Successfully loaded {} keys from HDD.", global_db.len());
            } else { println!("⚠️ Failed to deserialize {}. Starting fresh.", dump_file); }
        }
    }

    if args.statistics {
        crate::telemetry::spawn_telemetry_thread(Arc::clone(&global_db));
    }

    crate::accelerator::start_websocket_server(args.ws_port, num_workers);

    if let Some(hours) = args.hdd {
        let hdd_db = Arc::clone(&global_db);
        let dump_path = dump_file.to_string();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(hours * 3600));
            println!("💾 [PERSISTENCE] Initiating HDD Snapshot...");
            let mut snapshot = Vec::with_capacity(hdd_db.len());
            for entry in hdd_db.iter() { snapshot.push((entry.key().clone(), entry.value().clone())); }
            if let Ok(encoded) = bincode::serialize(&snapshot) {
                if fs::write(&dump_path, encoded).is_ok() {
                    println!("✅ [PERSISTENCE] Successfully saved {} keys to {}.", snapshot.len(), dump_path);
                } else { println!("❌ [PERSISTENCE] Failed to write {} to disk.", dump_path); }
            }
        });
    }

    for i in 0..num_workers {
        let model = Arc::clone(&model_base);
        let port = args.port;
        let bind_addr = args.bind.clone();
        #[cfg(unix)]
        let uds_listener_clone = Arc::clone(&uds_listener_base);
        let db = Arc::clone(&global_db);
        let vector_db_clone = Arc::clone(&vector_db);
        
        threads.push(std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async move {
                let addr: SocketAddr = format!("{}:{}", bind_addr, port).parse().unwrap();
                let socket = Socket::new(Domain::for_address(addr), Type::STREAM, Some(Protocol::TCP)).unwrap();
                socket.set_reuse_address(true).expect("failed to set reuse address");
                #[cfg(unix)]
                socket.set_reuse_port(true).expect("failed to set reuse port");
                socket.bind(&SockAddr::from(addr)).expect("failed to bind");
                socket.listen(4096).expect("failed to listen");
                socket.set_nonblocking(true).unwrap();
                let tcp_listener = TcpListener::from_std(socket.into()).unwrap();
                
                #[cfg(unix)]
                let uds_listener = UnixListener::from_std(uds_listener_clone.as_ref().try_clone().unwrap()).unwrap();
                
                println!("🌐 Worker {} online", i);
                
                let db_for_tcp = Arc::clone(&db);
                let model_for_tcp = Arc::clone(&model);
                let vector_db_for_tcp = Arc::clone(&vector_db_clone);

                #[cfg(unix)]
                {
                    let db_for_uds = Arc::clone(&db);
                    let model_for_uds = Arc::clone(&model);
                    let vector_db_for_uds = Arc::clone(&vector_db_clone);

                    tokio::spawn(async move {
                        loop {
                            if let Ok((s, _)) = uds_listener.accept().await {
                                let local_db = Arc::clone(&db_for_uds);
                                let local_model = Arc::clone(&model_for_uds);
                                let local_vector_db = Arc::clone(&vector_db_for_uds);
                                tokio::spawn(async move {
                                    let _ = handle_connection(s, local_db, local_model, local_vector_db).await;
                                });
                            }
                        }
                    });
                }

                loop {
                    if let Ok((s, _)) = tcp_listener.accept().await {
                        let _ = s.set_nodelay(true);
                        let local_db = Arc::clone(&db_for_tcp);
                        let local_model = Arc::clone(&model_for_tcp);
                        let local_vector_db = Arc::clone(&vector_db_for_tcp);
                        tokio::spawn(async move {
                            let _ = handle_connection(s, local_db, local_model, local_vector_db).await;
                        });
                    }
                }
            });
        }));
    }

    for t in threads { t.join().unwrap(); }
    Ok(())
}

async fn handle_connection<S>(mut socket: S, db: Arc<DashMap<AirKey, NeuralValue, fxhash::FxBuildHasher>>, model: Arc<McnnModel>, vector_db: Arc<crate::vector_store::SharedVectorStore>) -> Result<(), Box<dyn std::error::Error>> 
where S: AsyncReadExt + AsyncWriteExt + Unpin 
{
    let mut read_buffer = BytesMut::with_capacity(1024 * 1024 * 1024); // 1GB Buffer for Massive Blobs
    let mut write_buffer = BytesMut::with_capacity(1024 * 1024 * 1024);

    let is_lite = model.worker_limit <= 2;
    let mut tokens: f64 = 2000.0;
    let mut last_refill = std::time::Instant::now();

    loop {
        let n = socket.read_buf(&mut read_buffer).await?;
        if n == 0 { return Ok(()); }
        let mut pos = 0;
        while pos < read_buffer.len() {
            if read_buffer[pos] == b'*' {
                if let Some((cmd, consumed)) = parse_command_fast(&read_buffer[pos..]) {
                    if is_lite {
                        let now = std::time::Instant::now();
                        let elapsed = now.duration_since(last_refill).as_secs_f64();
                        last_refill = now;
                        tokens = (tokens + elapsed * 200000.0).min(2000.0);
                        if tokens < 1.0 {
                            let sleep_secs = (1.0 - tokens) / 200000.0;
                            tokio::time::sleep(std::time::Duration::from_secs_f64(sleep_secs)).await;
                            tokens = 0.0;
                        } else {
                            tokens -= 1.0;
                        }
                    }
                    let current_pos = pos + consumed;
                    match cmd {
                        Command::Ping => { write_buffer.extend_from_slice(make_pong()); pos = current_pos; }
                        Command::Set(key, value) => {
                            let (best_idx, chunks) = model.forward_cascade(value);
                            let nv = if chunks.len() == 1 {
                                NeuralValue::Single { best_idx: best_idx as u16, data: ChunkData(chunks[0]), len: value.len() }
                            } else {
                                NeuralValue::Multi { original_len: value.len(), best_idx: best_idx as u16, chunks: chunks.into_iter().map(ChunkData).collect() }
                            };
                            db.insert(AirKey::from_bytes(key.as_bytes()), nv);
                            crate::accelerator::broadcast_update(key, value);
                            write_buffer.extend_from_slice(RESP_OK);
                            pos = current_pos;
                        }
                        Command::Get(key) => {
                            if let Some(entry) = db.get(&AirKey::from_bytes(key.as_bytes())) {
                                let nv = entry.value();
                                match nv {
                                    NeuralValue::Single { best_idx, data, len } => {
                                        let full_data = model.reconstruct_cascade(*best_idx as usize, &[*data], *len);
                                        write_buffer.extend_from_slice(format!("${}\r\n", len).as_bytes());
                                        write_buffer.extend_from_slice(&full_data); write_buffer.extend_from_slice(RESP_CRLF);
                                    }
                                    NeuralValue::Multi { original_len, best_idx, chunks } => {
                                        let full_data = model.reconstruct_cascade(*best_idx as usize, chunks, *original_len);
                                        write_buffer.extend_from_slice(format!("${}\r\n", original_len).as_bytes());
                                        write_buffer.extend_from_slice(&full_data); write_buffer.extend_from_slice(RESP_CRLF);
                                    }
                                    NeuralValue::FlatHash(_) => {
                                        write_buffer.extend_from_slice(b"-WRONGTYPE Operation against a key holding the wrong kind of value\r\n");
                                    }
                                }
                            } else { write_buffer.extend_from_slice(RESP_NULL); }
                            pos = current_pos;
                        }
                        Command::Del(key) => {
                            let count = if db.remove(&AirKey::from_bytes(key.as_bytes())).is_some() { 1 } else { 0 };
                            write_buffer.extend_from_slice(&make_int(count));
                            pos = current_pos;
                        }
                        Command::Exists(key) => {
                            let count = if db.contains_key(&AirKey::from_bytes(key.as_bytes())) { 1 } else { 0 };
                            write_buffer.extend_from_slice(&make_int(count));
                            pos = current_pos;
                        }
                        Command::Select(_) => { write_buffer.extend_from_slice(RESP_OK); pos = current_pos; }
                        Command::FlushDb => { db.clear(); write_buffer.extend_from_slice(RESP_OK); pos = current_pos; }
                        Command::Info => {
                            let info = "redis_version:3.0.0\r\nomega_drive_version:3.0.0\r\nrole:master\r\n";
                            write_buffer.extend_from_slice(&make_bulk_string(info.as_bytes()));
                            pos = current_pos;
                        }
                        Command::MSetHeader(num_args) => {
                            let num_pairs = num_args / 2;
                            let mut temp_pos = current_pos;
                            let mut success = true;
                            let mut batch = Vec::with_capacity(num_pairs);
                            for _ in 0..num_pairs {
                                if let Some((key_raw, k_len)) = crate::resp::parse_bulk_str(&read_buffer, temp_pos) {
                                    temp_pos += k_len;
                                    if let Some((val, v_len)) = crate::resp::parse_bulk_str(&read_buffer, temp_pos) {
                                        temp_pos += v_len;
                                        batch.push((key_raw, val));
                                    } else { success = false; break; }
                                } else { success = false; break; }
                            }
                            if success {
                                for (key_raw, val) in batch {
                                    let (best_idx, chunks) = model.forward_raw(val);
                                    let nv = if chunks.len() == 1 {
                                        NeuralValue::Single { best_idx: best_idx as u16, data: ChunkData(chunks[0]), len: val.len() }
                                    } else {
                                        NeuralValue::Multi { original_len: val.len(), best_idx: best_idx as u16, chunks: chunks.into_iter().map(ChunkData).collect() }
                                    };
                                    db.insert(AirKey::from_bytes(key_raw), nv);
                                    if let Ok(key_str) = std::str::from_utf8(key_raw) {
                                        crate::accelerator::broadcast_update(key_str, val);
                                    }
                                }
                                write_buffer.extend_from_slice(RESP_OK); pos = temp_pos;
                            } else { break; }
                        }
                        Command::MGetHeader(num_args) => {
                            let mut temp_pos = current_pos;
                            let mut keys = Vec::with_capacity(num_args);
                            let mut success = true;
                            for _ in 0..num_args {
                                if let Some((key_raw, k_len)) = crate::resp::parse_bulk_str(&read_buffer, temp_pos) {
                                    temp_pos += k_len; keys.push(key_raw);
                                } else { success = false; break; }
                            }
                            if success {
                                write_buffer.extend_from_slice(format!("*{}\r\n", num_args).as_bytes());
                                for key_raw in keys {
                                    if let Some(entry) = db.get(&AirKey::from_bytes(key_raw)) {
                                        let nv = entry.value();
                                        let (full_data, orig_len) = match nv {
                                            NeuralValue::Single { best_idx, data, len } => (model.reconstruct(*best_idx as usize, &data.0)[..*len].to_vec(), *len),
                                            NeuralValue::Multi { original_len, best_idx, chunks } => (model.reconstruct_raw(*best_idx as usize, chunks, *original_len), *original_len),
                                            NeuralValue::FlatHash(_) => (Vec::new(), 0),
                                        };
                                        if orig_len > 0 {
                                            write_buffer.extend_from_slice(format!("${}\r\n", orig_len).as_bytes());
                                            write_buffer.extend_from_slice(&full_data); write_buffer.extend_from_slice(RESP_CRLF);
                                        } else {
                                            write_buffer.extend_from_slice(RESP_NULL);
                                        }
                                    } else { write_buffer.extend_from_slice(RESP_NULL); }
                                }
                                pos = temp_pos;
                            } else { break; }
                        }
                        Command::HmSet(key, fields) => {
                            let key_obj = AirKey::from_bytes(key.as_bytes());
                            let new_buf = if let Some(entry) = db.get_mut(&key_obj) {
                                let old_val = entry.value();
                                match old_val {
                                    NeuralValue::FlatHash(old_buf) => {
                                        hash_engine::FlatHash::merge(old_buf, &fields)
                                    }
                                    _ => {
                                        hash_engine::FlatHash::serialize(&fields)
                                    }
                                }
                            } else {
                                hash_engine::FlatHash::serialize(&fields)
                            };
                            db.insert(key_obj, NeuralValue::FlatHash(new_buf));
                            write_buffer.extend_from_slice(RESP_OK);
                            pos = current_pos;
                        }
                        Command::HGetAll(key) => {
                            if let Some(entry) = db.get(&AirKey::from_bytes(key.as_bytes())) {
                                let nv = entry.value();
                                match nv {
                                    NeuralValue::FlatHash(buf) => {
                                        hash_engine::FlatHash::deserialize_into_resp(buf, &mut write_buffer);
                                    }
                                    _ => {
                                        write_buffer.extend_from_slice(b"-WRONGTYPE Operation against a key holding the wrong kind of value\r\n");
                                    }
                                }
                            } else {
                                write_buffer.extend_from_slice(b"*0\r\n");
                            }
                            pos = current_pos;
                        }
                        Command::ZAdd(_) => {
                            write_buffer.extend_from_slice(b":1\r\n");
                            pos = current_pos;
                        }
                        Command::VAdd(key, floats) => {
                            let bit_vec = vector_db.quantize(&floats);
                            vector_db.add(AirKey::from_bytes(key.as_bytes()), bit_vec, floats);
                            write_buffer.extend_from_slice(RESP_OK);
                            pos = current_pos;
                        }
                        Command::VSearch(k, floats) => {
                            let query_bit_vec = vector_db.quantize(&floats);
                            let results = vector_db.search_reranked(&query_bit_vec, &floats, k);
                            
                            // Return flat array: [key_1, dist_1, key_2, dist_2, ...]
                            write_buffer.extend_from_slice(format!("*{}\r\n", results.len() * 2).as_bytes());
                            for (key, score) in results {
                                let key_bytes = key.as_bytes();
                                write_buffer.extend_from_slice(format!("${}\r\n", key_bytes.len()).as_bytes());
                                write_buffer.extend_from_slice(key_bytes);
                                write_buffer.extend_from_slice(RESP_CRLF);
                                let dist_int = (score * 10000.0) as i32;
                                write_buffer.extend_from_slice(format!(":{}\r\n", dist_int).as_bytes());
                            }
                            pos = current_pos;
                        }
                        Command::Unknown(_) => {
                            // Some clients send 'COMMAND' - we return an empty array to indicate compatibility
                            write_buffer.extend_from_slice(b"*0\r\n");
                            pos = current_pos;
                        }
                        _ => { write_buffer.extend_from_slice(b"-ERR unhandled\r\n"); pos = current_pos; }
                    }
                } else { break; }
            } else {
                let mut end = pos; while end < read_buffer.len() && read_buffer[end] != b'\n' { end += 1; }
                if end < read_buffer.len() { pos = end + 1; } else { break; }
            }
            if write_buffer.len() > 524288 { socket.write_all(&write_buffer).await?; write_buffer.clear(); }
        }
        read_buffer.advance(pos);
        if !write_buffer.is_empty() { socket.write_all(&write_buffer).await?; write_buffer.clear(); }
    }
}
