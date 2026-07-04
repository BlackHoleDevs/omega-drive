use std::time::Instant;
use rand::Rng;
use tokio::net::TcpStream;
use tokio::io::{AsyncWriteExt, AsyncReadExt};

const DIM: usize = 1536;
const NUM_BLOCKS: usize = DIM / 64; // 24 u64s
const NUM_VECTORS: usize = 50000;
const NUM_QUERIES: usize = 100;

fn main() {
    println!("🧪 OMEGA DRIVE v2.0 - VECTOR SEARCH BENCHMARK SUITE");
    println!("==================================================");
    println!("Configuration:");
    println!("  - Vector Dimensions: {}", DIM);
    println!("  - Database Size:     {} vectors", NUM_VECTORS);
    println!("  - Search Queries:    {}", NUM_QUERIES);
    println!("==================================================\n");

    let mut rng = rand::thread_rng();

    // 1. Generate clustered dataset to simulate realistic semantic embeddings (e.g. OpenAI/Cohere)
    println!("Generating clustered semantic vectors...");
    let num_clusters = 200;
    let mut clusters = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let c: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
        clusters.push(c);
    }

    let raw_float_db = generate_clustered_vectors(NUM_VECTORS, DIM, &clusters, 0.15);
    let raw_query_floats = generate_clustered_vectors(NUM_QUERIES, DIM, &clusters, 0.15);

    // 1.1 Pre-normalize vectors for standard production-grade Cosine Similarity (Dot Product baseline)
    println!("Pre-normalizing vectors for fair Cosine (dot-product) baseline...");
    let mut float_db = raw_float_db.clone();
    for v in &mut float_db {
        normalize_vector(v);
    }
    let mut query_floats = raw_query_floats.clone();
    for q in &mut query_floats {
        normalize_vector(q);
    }

    // Convert databases to flat contiguous memory layout (ideal for L1/L2 cache prefetching)
    let mut flat_float_db = Vec::with_capacity(NUM_VECTORS * DIM);
    for v in &float_db {
        flat_float_db.extend_from_slice(v);
    }

    // 1.2 Binary vectors (Omega Drive)
    println!("Quantizing vectors to 1-bit Binary representation...");
    let mut flat_binary_db = Vec::with_capacity(NUM_VECTORS * NUM_BLOCKS);
    for v in &raw_float_db {
        flat_binary_db.extend_from_slice(&quantize_vector(v));
    }
    
    let query_binaries: Vec<Vec<u64>> = raw_query_floats.iter().map(|q| quantize_vector(q)).collect();

    println!("Dataset generated.");
    let float_mem = (NUM_VECTORS * DIM * 4) as f64 / 1024.0 / 1024.0;
    let bin_mem = (NUM_VECTORS * NUM_BLOCKS * 8) as f64 / 1024.0 / 1024.0;
    println!("🧠 Memory Footprint Comparison:");
    println!("   - Standard Flat Float Database: {:.2} MB", float_mem);
    println!("   - Omega Flat Binary Vector DB:  {:.2} MB (Save: {:.1}x memory)", bin_mem, float_mem / bin_mem);
    println!("--------------------------------------------------\n");

    // 2. Benchmark Standard Float Cosine Similarity (Highly optimized via Dot Product)
    println!("Running Standard Float Cosine Similarity Benchmark (dot product)...");
    let start_float = Instant::now();
    let mut dummy_sum_float = 0.0;
    for q in &query_floats {
        let mut best_score = -1.0;
        for idx in 0..NUM_VECTORS {
            let offset = idx * DIM;
            let candidate = &flat_float_db[offset .. offset + DIM];
            let score = dot_product(q, candidate);
            if score > best_score {
                best_score = score;
            }
        }
        dummy_sum_float += best_score;
    }
    let float_duration = start_float.elapsed();
    let avg_float_latency = float_duration / NUM_QUERIES as u32;
    println!("✅ Standard Float Search Completed.");
    println!("   - Total Time: {:?}", float_duration);
    println!("   - Average Query Latency: {:?}", avg_float_latency);
    println!("--------------------------------------------------\n");

    // 3. Benchmark Omega Binary XOR + POPCNT (No Reranking)
    println!("Running Omega Binary (XOR + POPCNT) Benchmark (No Reranking)...");
    let start_bin = Instant::now();
    let mut dummy_sum_bin = 0;
    for q in &query_binaries {
        let mut best_dist = u32::MAX;
        for idx in 0..NUM_VECTORS {
            let offset = idx * NUM_BLOCKS;
            let candidate = &flat_binary_db[offset .. offset + NUM_BLOCKS];
            let mut dist = 0u32;
            for j in 0..NUM_BLOCKS {
                dist += (q[j] ^ candidate[j]).count_ones();
            }
            if dist < best_dist {
                best_dist = dist;
            }
        }
        dummy_sum_bin += best_dist;
    }
    let bin_duration = start_bin.elapsed();
    let avg_bin_latency = bin_duration / NUM_QUERIES as u32;
    println!("✅ Omega Binary Search Completed.");
    println!("   - Total Time: {:?}", bin_duration);
    println!("   - Average Query Latency: {:?}", avg_bin_latency);
    println!("--------------------------------------------------\n");

    // 4. Benchmark Reranked Binary Search (Coarse Filter Top 1000 -> Float Rerank Top 100)
    println!("Running Reranked Binary Search (Filter Top 1000 -> Rerank Top 100)...");
    let start_rerank = Instant::now();
    let mut dummy_sum_rerank = 0.0;
    for i in 0..NUM_QUERIES {
        let q_bin = &query_binaries[i];
        let q_float = &query_floats[i];
        let results = get_top_k_binary_reranked(q_bin, q_float, &flat_binary_db, &flat_float_db, NUM_VECTORS, 1000, 100);
        if !results.is_empty() {
            dummy_sum_rerank += flat_float_db[results[0] * DIM];
        }
    }
    let rerank_duration = start_rerank.elapsed();
    let avg_rerank_latency = rerank_duration / NUM_QUERIES as u32;
    println!("✅ Reranked Binary Search Completed.");
    println!("   - Total Time: {:?}", rerank_duration);
    println!("   - Average Query Latency: {:?}", avg_rerank_latency);
    println!("--------------------------------------------------\n");

    // 5. Calculate Recall
    println!("Evaluating Recall (Hamming Distance vs Reranked vs Ground-Truth Cosine)...");
    let mut recall_10_sum = 0.0;
    let mut recall_100_sum = 0.0;
    let mut rerank_recall_10_sum = 0.0;
    let mut rerank_recall_100_sum = 0.0;

    for i in 0..NUM_QUERIES {
        let q_float = &query_floats[i];
        let q_bin = &query_binaries[i];

        let top_10_float = get_top_k_float(q_float, &flat_float_db, NUM_VECTORS, 10);
        let top_100_float = get_top_k_float(q_float, &flat_float_db, NUM_VECTORS, 100);

        let top_10_bin = get_top_k_binary(q_bin, &flat_binary_db, NUM_VECTORS, 10);
        let top_100_bin = get_top_k_binary(q_bin, &flat_binary_db, NUM_VECTORS, 100);

        let top_10_rerank = get_top_k_binary_reranked(q_bin, q_float, &flat_binary_db, &flat_float_db, NUM_VECTORS, 500, 10);
        let top_100_rerank = get_top_k_binary_reranked(q_bin, q_float, &flat_binary_db, &flat_float_db, NUM_VECTORS, 1000, 100);

        recall_10_sum += calculate_recall(&top_10_float, &top_10_bin);
        recall_100_sum += calculate_recall(&top_100_float, &top_100_bin);

        rerank_recall_10_sum += calculate_recall(&top_10_float, &top_10_rerank);
        rerank_recall_100_sum += calculate_recall(&top_100_float, &top_100_rerank);
    }
    let recall_10 = recall_10_sum / NUM_QUERIES as f64;
    let recall_100 = recall_100_sum / NUM_QUERIES as f64;
    let rerank_recall_10 = rerank_recall_10_sum / NUM_QUERIES as f64;
    let rerank_recall_100 = rerank_recall_100_sum / NUM_QUERIES as f64;

    println!("✅ Recall Analysis:");
    println!("   - Raw Binary Recall@10:       {:.2}%", recall_10 * 100.0);
    println!("   - Raw Binary Recall@100:      {:.2}%", recall_100 * 100.0);
    println!("   - Reranked Binary Recall@10:  {:.2}% (Rerank 500)", rerank_recall_10 * 100.0);
    println!("   - Reranked Binary Recall@100: {:.2}% (Rerank 1000)", rerank_recall_100 * 100.0);
    println!("--------------------------------------------------\n");

    // 6. Run Scaling Test (including 1,000,000 vectors!)
    println!("📊 RUNNING DATASET SCALING TESTS (Up to 1,000,000 vectors)...");
    let sizes = [10000, 50000, 100000, 250000, 1000000];
    
    println!("+-------------+------------------+------------------+--------------------+-------------------+---------------+");
    println!("|  DB Size    |  Avg Float Lat.  |  Avg Raw Bin Lat.|  Avg Reranked Lat. |  Reranked Speedup | Reranked R@100|");
    println!("+-------------+------------------+------------------+--------------------+-------------------+---------------+");

    for &size in &sizes {
        // Generate dataset of target size in flat contiguous format
        let temp_raw_db = generate_clustered_vectors(size, DIM, &clusters, 0.15);
        let mut temp_flat_float = Vec::with_capacity(size * DIM);
        for v in &temp_raw_db {
            let mut normalized = v.clone();
            normalize_vector(&mut normalized);
            temp_flat_float.extend_from_slice(&normalized);
        }

        let mut temp_flat_bin = Vec::with_capacity(size * NUM_BLOCKS);
        for v in &temp_raw_db {
            temp_flat_bin.extend_from_slice(&quantize_vector(v));
        }

        // Dynamically scale coarse-filter candidate pool size (1% of N, minimum 1000)
        let r_size = (size / 100).max(1000);

        // Benchmark float
        let mut d_sum_float = 0.0;
        let start_t = Instant::now();
        for q in &query_floats {
            let mut best_score = -1.0;
            for idx in 0..size {
                let offset = idx * DIM;
                let candidate = &temp_flat_float[offset .. offset + DIM];
                let score = dot_product(q, candidate);
                if score > best_score {
                    best_score = score;
                }
            }
            d_sum_float += best_score;
        }
        let t_float = start_t.elapsed() / NUM_QUERIES as u32;
        std::hint::black_box(d_sum_float);

        // Benchmark raw binary
        let mut d_sum_bin = 0;
        let start_t_bin = Instant::now();
        for q in &query_binaries {
            let mut best_dist = u32::MAX;
            for idx in 0..size {
                let offset = idx * NUM_BLOCKS;
                let candidate = &temp_flat_bin[offset .. offset + NUM_BLOCKS];
                let mut dist = 0u32;
                for j in 0..NUM_BLOCKS {
                    dist += (q[j] ^ candidate[j]).count_ones();
                }
                if dist < best_dist {
                    best_dist = dist;
                }
            }
            d_sum_bin += best_dist;
        }
        let t_bin = start_t_bin.elapsed() / NUM_QUERIES as u32;
        std::hint::black_box(d_sum_bin);

        // Benchmark Reranked binary
        let mut d_sum_rerank = 0.0;
        let start_t_rerank = Instant::now();
        for i in 0..NUM_QUERIES {
            let q_bin = &query_binaries[i];
            let q_float = &query_floats[i];
            let results = get_top_k_binary_reranked(q_bin, q_float, &temp_flat_bin, &temp_flat_float, size, r_size, 100);
            if !results.is_empty() {
                d_sum_rerank += temp_flat_float[results[0] * DIM];
            }
        }
        let t_rerank = start_t_rerank.elapsed() / NUM_QUERIES as u32;
        std::hint::black_box(d_sum_rerank);

        // Reranked Recall@100 for this size (limit evaluation queries to 10 for speed at 1M)
        let mut r100_sum = 0.0;
        let eval_queries = 10.min(NUM_QUERIES);
        for i in 0..eval_queries {
            let top_f = get_top_k_float(&query_floats[i], &temp_flat_float, size, 100);
            let top_b = get_top_k_binary_reranked(&query_binaries[i], &query_floats[i], &temp_flat_bin, &temp_flat_float, size, r_size, 100);
            r100_sum += calculate_recall(&top_f, &top_b);
        }
        let r100 = r100_sum / eval_queries as f64;

        let scale_speedup = t_float.as_nanos() as f64 / t_rerank.as_nanos() as f64;

        println!(
            "| {:<11} | {:<16?} | {:<16?} | {:<18?} | {:<17.2}x | {:<13.2}% |",
            size,
            t_float,
            t_bin,
            t_rerank,
            scale_speedup,
            r100 * 100.0
        );
    }
    println!("+-------------+------------------+------------------+--------------------+-------------------+---------------+");
    println!("==================================================\n");

    // 7. Summary
    let raw_speedup = float_duration.as_nanos() as f64 / bin_duration.as_nanos() as f64;
    let rerank_speedup = float_duration.as_nanos() as f64 / rerank_duration.as_nanos() as f64;
    println!("🏆 RESULTS SUMMARY (for {} vectors):", NUM_VECTORS);
    println!("==================================================");
    println!("  ⚡ Raw Binary Speedup:      {:.2}x FASTER", raw_speedup);
    println!("  ⚡ Reranked Binary Speedup: {:.2}x FASTER", rerank_speedup);
    println!("  🧠 Memory Savings:          {:.2}x REDUCED FOOTPRINT", float_mem / bin_mem);
    println!("  🎯 Raw Recall@100:          {:.2}%", recall_100 * 100.0);
    println!("  🎯 Reranked Recall@100:     {:.2}%", rerank_recall_100 * 100.0);
    println!("==================================================");
    println!("(Verification check: sum_float={:.2}, sum_bin={}, sum_rerank={:.2})", dummy_sum_float, dummy_sum_bin, dummy_sum_rerank);
    
    println!("\nNow trying network integration test with running Omega Drive server...");

    // Spawn async network test
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async_network_test());

    println!("\n==================================================");
    println!("🧪 BONUS: WEBSOCKET FRAME DEMASKING AVX2 BENCHMARK");
    println!("==================================================");
    let payload_size = 64 * 1024; // 64 KB
    let iterations = 20000;
    println!("Payload Size:   {} KB", payload_size / 1024);
    println!("Iterations:     {}", iterations);

    let mut test_payload = vec![0u8; payload_size];
    let mut rng_payload = rand::thread_rng();
    for i in 0..payload_size {
        test_payload[i] = rng_payload.gen::<u8>();
    }
    let mask = [0xAA, 0xBB, 0xCC, 0xDD];

    // Fallback benchmark
    let mut payload_fallback = test_payload.clone();
    let start_fallback = Instant::now();
    for _ in 0..iterations {
        demask_fallback(&mut payload_fallback, mask);
    }
    let fallback_duration = start_fallback.elapsed();
    println!("   - Fallback (Byte-by-Byte): {:?}", fallback_duration);

    // AVX2 benchmark
    let mut payload_avx2 = test_payload.clone();
    let start_avx2 = Instant::now();
    for _ in 0..iterations {
        demask_avx2(&mut payload_avx2, mask);
    }
    let avx2_duration = start_avx2.elapsed();
    println!("   - AVX2 SIMD:              {:?}", avx2_duration);
    
    let ws_speedup = fallback_duration.as_nanos() as f64 / avx2_duration.as_nanos() as f64;
    println!("  ⚡ AVX2 Demasking Speedup: {:.2}x FASTER", ws_speedup);
    println!("==================================================");
}

fn generate_clustered_vectors(num_vectors: usize, _dim: usize, clusters: &[Vec<f32>], noise: f32) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut db = Vec::with_capacity(num_vectors);
    for _ in 0..num_vectors {
        let cluster_idx = rng.gen_range(0..clusters.len());
        let mut vec = clusters[cluster_idx].clone();
        for val in vec.iter_mut() {
            *val += rng.gen_range(-noise..noise);
        }
        db.push(vec);
    }
    db
}

fn quantize_vector(floats: &[f32]) -> Vec<u64> {
    let mut packed = vec![0u8; NUM_BLOCKS * 8];
    for i in 0..floats.len() {
        if floats[i] >= 0.0 {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            packed[byte_idx] |= 1 << bit_idx;
        }
    }
    let mut u64_data = vec![0u64; NUM_BLOCKS];
    for i in 0..NUM_BLOCKS {
        let start = i * 8;
        let mut val = 0u64;
        for j in 0..8 {
            val |= (packed[start + j] as u64) << (j * 8);
        }
        u64_data[i] = val;
    }
    u64_data
}

fn normalize_vector(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 0.0 {
        for val in v.iter_mut() {
            *val /= norm;
        }
    }
}

// Manually unrolled 8x accumulator loop to allow AVX2/FMA auto-vectorization
// ignoring IEEE 754 sum reordering blocks to unleash hardware potential
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

fn get_top_k_float(query: &[f32], db: &[f32], num_vectors: usize, k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, f32)> = (0..num_vectors)
        .map(|idx| {
            let offset = idx * DIM;
            let candidate = &db[offset .. offset + DIM];
            (idx, dot_product(query, candidate))
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.iter().take(k).map(|&(idx, _)| idx).collect()
}

fn get_top_k_binary(query: &[u64], db: &[u64], num_vectors: usize, k: usize) -> Vec<usize> {
    let mut scores: Vec<(usize, u32)> = (0..num_vectors)
        .map(|idx| {
            let offset = idx * NUM_BLOCKS;
            let candidate = &db[offset .. offset + NUM_BLOCKS];
            let mut dist = 0u32;
            for j in 0..NUM_BLOCKS {
                dist += (query[j] ^ candidate[j]).count_ones();
            }
            (idx, dist)
        })
        .collect();
    scores.sort_unstable_by_key(|&(_, dist)| dist);
    scores.iter().take(k).map(|&(idx, _)| idx).collect()
}

fn get_top_k_binary_reranked(
    query_bin: &[u64],
    query_float: &[f32],
    binary_db: &[u64],
    float_db: &[f32],
    num_vectors: usize,
    r: usize,
    k: usize,
) -> Vec<usize> {
    let mut scores: Vec<(usize, u32)> = (0..num_vectors)
        .map(|idx| {
            let offset = idx * NUM_BLOCKS;
            let candidate = &binary_db[offset .. offset + NUM_BLOCKS];
            let mut dist = 0u32;
            for j in 0..NUM_BLOCKS {
                dist += (query_bin[j] ^ candidate[j]).count_ones();
            }
            (idx, dist)
        })
        .collect();

    let r_clamped = r.min(num_vectors);
    if r_clamped > 0 {
        scores.select_nth_unstable_by_key(r_clamped - 1, |&(_, dist)| dist);
        scores.truncate(r_clamped);
    }

    let mut reranked: Vec<(usize, f32)> = scores.iter()
        .map(|&(idx, _)| {
            let offset = idx * DIM;
            let candidate = &float_db[offset .. offset + DIM];
            let score = dot_product(query_float, candidate);
            (idx, score)
        })
        .collect();

    reranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    reranked.iter().take(k).map(|&(idx, _)| idx).collect()
}

fn calculate_recall(float_top: &[usize], binary_top: &[usize]) -> f64 {
    let mut matches = 0;
    for &idx in binary_top {
        if float_top.contains(&idx) {
            matches += 1;
        }
    }
    matches as f64 / float_top.len() as f64
}

async fn async_network_test() {
    let addr = "127.0.0.1:6380";
    println!("Connecting to Omega Drive on {}...", addr);
    
    let mut stream = match TcpStream::connect(addr).await {
        Ok(s) => s,
        Err(_) => {
            println!("⚠️ Omega Drive server is not running on {}. Skipping live network test.", addr);
            return;
        }
    };
    println!("✅ Connected to Omega Drive!");

    let mut rng = rand::thread_rng();
    let mut test_vec: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1.0..1.0)).collect();
    normalize_vector(&mut test_vec);
    
    let key = "bench_vector_1";
    let mut cmd = format!("*1538\r\n$4\r\nVADD\r\n${}\r\n{}\r\n", key.len(), key);
    for val in &test_vec {
        let val_str = format!("{:.4}", val);
        cmd.push_str(&format!("${}\r\n{}\r\n", val_str.len(), val_str));
    }

    println!("Sending VADD to Omega Drive...");
    let start_write = Instant::now();
    if let Err(e) = stream.write_all(cmd.as_bytes()).await {
        println!("❌ Failed to write VADD command: {}", e);
        return;
    }
    
    let mut response = vec![0u8; 1024];
    match stream.read(&mut response).await {
        Ok(n) => {
            let resp_str = String::from_utf8_lossy(&response[..n]);
            println!("Response in {:?}: {}", start_write.elapsed(), resp_str.trim());
        }
        Err(e) => println!("❌ Failed to read response: {}", e),
    }

    let mut search_cmd = format!("*1538\r\n$7\r\nVSEARCH\r\n$1\r\n5\r\n");
    for val in &test_vec {
        let val_str = format!("{:.4}", val);
        search_cmd.push_str(&format!("${}\r\n{}\r\n", val_str.len(), val_str));
    }

    println!("Sending VSEARCH to Omega Drive...");
    let start_search = Instant::now();
    if let Err(e) = stream.write_all(search_cmd.as_bytes()).await {
        println!("❌ Failed to write VSEARCH command: {}", e);
        return;
    }

    let mut search_response = vec![0u8; 1024];
    match stream.read(&mut search_response).await {
        Ok(n) => {
            let resp_str = String::from_utf8_lossy(&search_response[..n]);
            println!("Search results in {:?}:\n{}", start_search.elapsed(), resp_str.trim());
        }
        Err(e) => println!("❌ Failed to read response: {}", e),
    }
}

fn demask_fallback(data: &mut [u8], mask: [u8; 4]) {
    for (i, byte) in data.iter_mut().enumerate() {
        *byte ^= mask[i % 4];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn demask_avx2_impl(data: &mut [u8], mask: [u8; 4]) {
    let len = data.len();
    let mask_u32 = u32::from_ne_bytes(mask);
    let mut i = 0;

    use std::arch::x86_64::*;
    let v_mask = _mm256_set1_epi32(mask_u32 as i32);

    while i + 32 <= len {
        let ptr = data.as_mut_ptr().add(i);
        let v_data = _mm256_loadu_si256(ptr as *const __m256i);
        let v_res = _mm256_xor_si256(v_data, v_mask);
        _mm256_storeu_si256(ptr as *mut __m256i, v_res);
        i += 32;
    }

    if i < len {
        demask_fallback(&mut data[i..], mask);
    }
}

fn demask_avx2(data: &mut [u8], mask: [u8; 4]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                demask_avx2_impl(data, mask);
            }
            return;
        }
    }
    demask_fallback(data, mask);
}
