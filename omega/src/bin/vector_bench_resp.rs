use std::net::TcpStream;
use std::io::{Read, Write, BufReader};
use std::time::Instant;
use std::sync::Arc;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use chacha20::cipher::{KeyIvInit, StreamCipher};
use chacha20::ChaCha20;

const PORT: u16 = 6380;
const DIM: usize = 1536;

fn format_vadd_cmd(key: &str, floats: &[f32]) -> Vec<u8> {
    let num_args = 2 + floats.len();
    let mut buf = Vec::new();
    buf.extend_from_slice(format!("*{}\r\n", num_args).as_bytes());
    
    // Command
    buf.extend_from_slice(b"$4\r\nVADD\r\n");
    
    // Key
    buf.extend_from_slice(format!("${}\r\n", key.len()).as_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(b"\r\n");
    
    // Floats
    for &f in floats {
        let f_str = format!("{:.6}", f);
        buf.extend_from_slice(format!("${}\r\n", f_str.len()).as_bytes());
        buf.extend_from_slice(f_str.as_bytes());
        buf.extend_from_slice(b"\r\n");
    }
    buf
}

fn format_vsearch_cmd(k: usize, floats: &[f32]) -> Vec<u8> {
    let num_args = 2 + floats.len();
    let mut buf = Vec::new();
    buf.extend_from_slice(format!("*{}\r\n", num_args).as_bytes());
    
    // Command
    buf.extend_from_slice(b"$7\r\nVSEARCH\r\n");
    
    // k
    let k_str = k.to_string();
    buf.extend_from_slice(format!("${}\r\n", k_str.len()).as_bytes());
    buf.extend_from_slice(k_str.as_bytes());
    buf.extend_from_slice(b"\r\n");
    
    // Floats
    for &f in floats {
        let f_str = format!("{:.6}", f);
        buf.extend_from_slice(format!("${}\r\n", f_str.len()).as_bytes());
        buf.extend_from_slice(f_str.as_bytes());
        buf.extend_from_slice(b"\r\n");
    }
    buf
}

fn read_line<R: Read>(reader: &mut R) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        reader.read_exact(&mut byte)?;
        buf.push(byte[0]);
        if buf.ends_with(b"\r\n") {
            break;
        }
    }
    Ok(buf)
}

fn read_vsearch_response<R: Read>(reader: &mut R) -> std::io::Result<Vec<(String, f32)>> {
    let header = read_line(reader)?;
    if !header.starts_with(b"*") {
        let err_msg = String::from_utf8_lossy(&header).to_string();
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Expected array, got: {}", err_msg)));
    }
    let count: usize = std::str::from_utf8(&header[1..header.len()-2]).unwrap().parse().unwrap();
    let mut results = Vec::new();
    let mut current_key = String::new();
    for _ in 0..count {
        let el_header = read_line(reader)?;
        if el_header.starts_with(b"$") {
            let len: usize = std::str::from_utf8(&el_header[1..el_header.len()-2]).unwrap().parse().unwrap();
            let mut data = vec![0u8; len + 2];
            reader.read_exact(&mut data)?;
            current_key = std::str::from_utf8(&data[..len]).unwrap().to_string();
        } else if el_header.starts_with(b":") {
            let val_int: i32 = std::str::from_utf8(&el_header[1..el_header.len()-2]).unwrap().parse().unwrap();
            let val = val_int as f32 / 10000.0;
            results.push((current_key.clone(), val));
        }
    }
    Ok(results)
}

fn generate_deterministic_embedding(text: &str) -> Vec<f32> {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash_result = hasher.finalize();

    let mut key = [0u8; 32];
    key.copy_from_slice(&hash_result);
    let iv = [0u8; 12];
    let mut cipher = ChaCha20::new(&key.into(), &iv.into());

    let mut buffer = vec![0u8; DIM * 4];
    cipher.apply_keystream(&mut buffer);

    let mut floats = vec![0.0f32; DIM];
    let mut norm_sq = 0.0f32;
    for i in 0..DIM {
        let val_u32 = u32::from_ne_bytes([
            buffer[i * 4],
            buffer[i * 4 + 1],
            buffer[i * 4 + 2],
            buffer[i * 4 + 3],
        ]);
        let val_f32 = (val_u32 as f32 / u32::MAX as f32) * 2.0 - 1.0;
        floats[i] = val_f32;
        norm_sq += val_f32 * val_f32;
    }

    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        for val in floats.iter_mut() {
            *val /= norm;
        }
    }
    floats
}

// Manually unrolled 8x accumulator loop for float dot product
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let mut sums = [0.0f32; 8];
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for j in 0..8 {
            sums[j] += ca[j] * cb[j];
        }
    }
    sums.iter().sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let target_vectors = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(100_000)
    } else {
        1_000_000
    };

    println!("📖 Pre-generating 2000 high-dimension base vectors...");
    let base_vectors: Vec<Vec<f32>> = (0..2000)
        .into_par_iter()
        .map(|i| {
            generate_deterministic_embedding(&format!("base_sentence_text_chunk_{}", i))
        })
        .collect();
    let base_vectors = Arc::new(base_vectors);

    let get_vector_mixed = |global_idx: usize, bases: &Arc<Vec<Vec<f32>>>| -> Vec<f32> {
        let base = &bases[global_idx % 2000];
        let mut noise_seed = global_idx;
        let mut res = Vec::with_capacity(DIM);
        for &val in base {
            noise_seed = (1103515245usize.wrapping_mul(noise_seed).wrapping_add(12345)) & 0x7fffffff;
            let noise = ((noise_seed as f32 / 2147483648.0) - 0.5) * 0.05;
            res.push(val + noise);
        }
        let mut norm_sq = 0.0f32;
        for &x in &res {
            norm_sq += x * x;
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for x in res.iter_mut() {
                *x /= norm;
            }
        }
        res
    };

    // 0. Clean DB
    {
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", PORT))
            .expect("Failed to connect to Omega Drive");
        stream.write_all(b"*1\r\n$7\r\nFLUSHDB\r\n").unwrap();
        let mut resp = [0u8; 5];
        stream.read_exact(&mut resp).unwrap();
        assert_eq!(&resp[..3], b"+OK", "FLUSHDB failed");
        println!("🧹 Database cleared via FLUSHDB command.");
    }

    let num_threads = num_cpus::get().min(16);
    
    // 1. Baseline Brute-Force float search
    println!("\n🏋️ Running single-core Brute-Force float search baseline ({} vectors, no quantization)...", target_vectors);
    let test_query = &base_vectors[0];
    let start_bf = Instant::now();
    let mut best_score = -1.0f32;
    let mut _best_idx = 0;
    
    for idx in 0..target_vectors {
        let candidate = &base_vectors[idx % 2000];
        let score = dot_product(test_query, candidate);
        if score > best_score {
            best_score = score;
            _best_idx = idx;
        }
    }
    let bf_duration = start_bf.elapsed();
    println!("⏱️  Single-core Brute-Force Float Baseline:");
    println!("   Scan Time:  {:.2} ms", bf_duration.as_secs_f64() * 1000.0);
    println!("   Throughput: {:.2} QPS (Best score: {:.4} at idx: {})", 1.0 / bf_duration.as_secs_f64(), best_score, _best_idx);

    // 2. Parallel network loading
    println!("\n🚀 Starting parallel loading of {} vectors into Omega Drive (port {}) using {} connections...", 
             target_vectors, PORT, num_threads);

    let start_load = Instant::now();
    let chunk_size = target_vectors / num_threads;
    
    let handles: Vec<_> = (0..num_threads).map(|thread_idx| {
        let bases_clone = Arc::clone(&base_vectors);
        std::thread::spawn(move || {
            let mut write_stream = TcpStream::connect(format!("127.0.0.1:{}", PORT))
                .expect("Failed to connect to Omega Drive");
            write_stream.set_nodelay(true).unwrap();
            let read_stream = write_stream.try_clone().unwrap();
            let mut reader = BufReader::new(read_stream);

            let start_idx = thread_idx * chunk_size;
            let end_idx = if thread_idx == num_threads - 1 { target_vectors } else { start_idx + chunk_size };

            let batch_size = 500;
            let mut buf = Vec::new();
            let mut count = 0;

            for idx in start_idx..end_idx {
                let vec = get_vector_mixed(idx, &bases_clone);
                let key = format!("vec_{}", idx);
                let cmd = format_vadd_cmd(&key, &vec);
                buf.extend_from_slice(&cmd);
                count += 1;

                if count >= batch_size || idx == end_idx - 1 {
                    write_stream.write_all(&buf).unwrap();
                    
                    // Read responses
                    for _ in 0..count {
                        let resp = read_line(&mut reader).unwrap();
                        assert!(resp.starts_with(b"+OK") || resp.starts_with(b"+"), "Unexpected response: {:?}", resp);
                    }
                    buf.clear();
                    count = 0;
                }
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    let load_duration = start_load.elapsed();
    println!("💾 Total load/quantization time over network: {:.2} seconds", load_duration.as_secs_f64());
    println!("   Throughput: {:.1} VADD/sec", target_vectors as f64 / load_duration.as_secs_f64());

    // Warmup VSEARCH
    let mut write_stream = TcpStream::connect(format!("127.0.0.1:{}", PORT))
        .expect("Failed to connect to Omega Drive");
    write_stream.set_nodelay(true).unwrap();
    let read_stream = write_stream.try_clone().unwrap();
    let mut reader = BufReader::new(read_stream);

    let warmup_vec = get_vector_mixed(999, &base_vectors);
    let warmup_cmd = format_vsearch_cmd(10, &warmup_vec);
    for _ in 0..10 {
        write_stream.write_all(&warmup_cmd).unwrap();
        read_vsearch_response(&mut reader).unwrap();
    }

    // 3. Sequential network query (latency baseline)
    println!("\n🔍 Running sequential VSEARCH benchmarks (latency measurement)...");
    let num_queries = 200;
    let start_seq = Instant::now();
    for q_idx in 0..num_queries {
        let q_vec = get_vector_mixed(target_vectors + q_idx, &base_vectors);
        let q_cmd = format_vsearch_cmd(10, &q_vec);
        write_stream.write_all(&q_cmd).unwrap();
        read_vsearch_response(&mut reader).unwrap();
    }
    let seq_duration = start_seq.elapsed() / num_queries as u32;
    println!("⏱️  Sequential Roundtrip Performance:");
    println!("   Avg Latency: {:.2} ms", seq_duration.as_secs_f64() * 1000.0);
    println!("   Throughput:  {:.1} QPS", 1.0 / seq_duration.as_secs_f64());

    // 4. Concurrent network query (throughput measurement)
    println!("\n🔍 Running concurrent VSEARCH benchmarks ({} parallel connections)...", num_threads);
    let start_concurrent = Instant::now();
    let num_queries_per_thread = 50;
    let total_queries = num_queries_per_thread * num_threads;

    let search_handles: Vec<_> = (0..num_threads).map(|thread_idx| {
        let bases_clone = Arc::clone(&base_vectors);
        std::thread::spawn(move || {
            let mut write_stream = TcpStream::connect(format!("127.0.0.1:{}", PORT))
                .expect("Failed to connect to Omega Drive");
            write_stream.set_nodelay(true).unwrap();
            let read_stream = write_stream.try_clone().unwrap();
            let mut reader = BufReader::new(read_stream);

            for q_idx in 0..num_queries_per_thread {
                let q_vec = get_vector_mixed(target_vectors + thread_idx * 1000 + q_idx, &bases_clone);
                let q_cmd = format_vsearch_cmd(10, &q_vec);
                write_stream.write_all(&q_cmd).unwrap();
                read_vsearch_response(&mut reader).unwrap();
            }
        })
    }).collect();

    for h in search_handles {
        h.join().unwrap();
    }

    let concurrent_duration = start_concurrent.elapsed();
    let avg_client_latency = (concurrent_duration.as_secs_f64() * 1000.0) / num_queries_per_thread as f64;
    let concurrent_qps = total_queries as f64 / concurrent_duration.as_secs_f64();

    println!("⏱️  Concurrent Roundtrip Performance:");
    println!("   Total Queries:          {}", total_queries);
    println!("   Avg Response Latency:   {:.2} ms", avg_client_latency);
    println!("   System Throughput:      {:.1} QPS", concurrent_qps);

    // Fetch and print top-10 for verification
    let q_vec = get_vector_mixed(target_vectors + 999, &base_vectors);
    let q_cmd = format_vsearch_cmd(10, &q_vec);
    write_stream.write_all(&q_cmd).unwrap();
    let results = read_vsearch_response(&mut reader).unwrap();

    println!("\n🎯 TOP-10 Nearest Neighbors from running Omega server (Verification):");
    for (i, (key, score)) in results.iter().enumerate() {
        println!("   {}. Score: {:.4} | Key: {}", i + 1, score, key);
    }
}
