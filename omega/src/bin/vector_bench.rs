use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use std::collections::BinaryHeap;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use chacha20::cipher::{KeyIvInit, StreamCipher};
use chacha20::ChaCha20;

#[derive(Clone)]
struct AirKey(String);

struct CoreShard {
    keys: Vec<AirKey>,
    data: Vec<u64>,
    float_data: Vec<f32>,
}

impl CoreShard {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            data: Vec::new(),
            float_data: Vec::new(),
        }
    }

    // Thread-local Hamming distance search using a Max-Heap to keep the smallest distances
    fn search(&self, query: &[u64], num_blocks: usize, k: usize) -> Vec<(usize, u32)> {
        // Max-heap stores (distance, index)
        // We want to pop the largest distance when size > k, so the heap contains the smallest distances.
        let mut heap = BinaryHeap::with_capacity(k + 1);
        let num_vectors = self.keys.len();
        
        for i in 0..num_vectors {
            let start = i * num_blocks;
            let candidate = &self.data[start..start + num_blocks];
            let mut dist = 0u32;
            for j in 0..num_blocks {
                dist += (candidate[j] ^ query[j]).count_ones();
            }
            
            heap.push((dist, i));
            if heap.len() > k {
                heap.pop();
            }
        }
        
        let mut results: Vec<(u32, usize)> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(dist, _)| dist);
        results.into_iter().map(|(dist, idx)| (idx, dist)).collect()
    }
}

// Wrapper for min-heap of float scores (to keep the largest scores)
#[derive(Eq, PartialEq)]
struct RerankCandidate {
    score_int: i32,
    shard_idx: usize,
    local_idx: usize,
}

impl Ord for RerankCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order so the smallest score is popped first (min-heap)
        other.score_int.cmp(&self.score_int)
    }
}

impl PartialOrd for RerankCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct ShardedVectorEngine {
    shards: Vec<CoreShard>,
    dim: usize,
    num_blocks: usize,
}

impl ShardedVectorEngine {
    fn new(dim: usize, num_shards: usize) -> Self {
        assert!(dim % 64 == 0);
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(CoreShard::new());
        }
        Self {
            shards,
            dim,
            num_blocks: dim / 64,
        }
    }

    fn add(&mut self, key: AirKey, bit_vector: Vec<u64>, float_vector: Vec<f32>, global_idx: usize) {
        let shard_idx = global_idx % self.shards.len();
        self.shards[shard_idx].keys.push(key);
        self.shards[shard_idx].data.extend_from_slice(&bit_vector);
        self.shards[shard_idx].float_data.extend_from_slice(&float_vector);
    }

    fn quantize(&self, floats: &[f32]) -> Vec<u64> {
        let mut u64_data = vec![0u64; self.num_blocks];
        for i in 0..self.dim {
            if floats[i] >= 0.0 {
                let block_idx = i / 64;
                let bit_idx = i % 64;
                u64_data[block_idx] |= 1 << bit_idx;
            }
        }
        u64_data
    }

    // Shared-nothing sharded parallel Hamming search
    fn search_sharded_parallel(&self, query_bin: &[u64], k: usize) -> Vec<(AirKey, u32)> {
        // 1. Scan each shard on its own thread in parallel
        let shard_results: Vec<Vec<(usize, u32)>> = self.shards
            .par_iter()
            .map(|shard| {
                shard.search(query_bin, self.num_blocks, k)
            })
            .collect();

        // 2. Merge local top-K results from all shards
        let mut merged = BinaryHeap::with_capacity(k + 1);
        for (shard_idx, shard_res) in shard_results.into_iter().enumerate() {
            for (local_idx, dist) in shard_res {
                merged.push((dist, shard_idx, local_idx));
                if merged.len() > k {
                    merged.pop();
                }
            }
        }

        let mut final_results: Vec<(u32, usize, usize)> = merged.into_iter().collect();
        final_results.sort_unstable_by_key(|&(dist, _, _)| dist);

        final_results.into_iter()
            .map(|(dist, shard_idx, local_idx)| {
                (self.shards[shard_idx].keys[local_idx].clone(), dist)
            })
            .collect()
    }

    // Shared-nothing sharded parallel Hamming search + Float Reranking
    fn search_sharded_reranked_parallel(&self, query_bin: &[u64], query_float: &[f32], k: usize) -> Vec<(AirKey, f32)> {
        let total_vectors: usize = self.shards.iter().map(|s| s.keys.len()).sum();
        let r = (total_vectors / 100).max(1000).min(total_vectors); // 1% candidates
        let r_per_shard = (r / self.shards.len()).max(k);

        // 1. Parallel Hamming distance scan to get local top R candidates per shard
        let shard_candidates: Vec<Vec<(usize, u32)>> = self.shards
            .par_iter()
            .map(|shard| {
                shard.search(query_bin, self.num_blocks, r_per_shard)
            })
            .collect();

        // 2. Parallel exact float reranking within each shard
        let shard_reranked: Vec<Vec<(usize, f32)>> = self.shards
            .par_iter()
            .zip(shard_candidates.par_iter())
            .map(|(shard, candidates)| {
                let mut heap = BinaryHeap::with_capacity(k + 1);
                for &(local_idx, _) in candidates {
                    let start = local_idx * self.dim;
                    let candidate = &shard.float_data[start..start + self.dim];
                    let score = dot_product(query_float, candidate);
                    let score_int = (score * 10_000_000.0) as i32;
                    
                    heap.push(RerankCandidate { score_int, shard_idx: 0, local_idx });
                    if heap.len() > k {
                        heap.pop();
                    }
                }
                heap.into_iter()
                    .map(|c| (c.local_idx, c.score_int as f32 / 10_000_000.0))
                    .collect()
            })
            .collect();

        // 3. Merge final candidates from all shards
        let mut final_heap = BinaryHeap::with_capacity(k + 1);
        for (shard_idx, shard_res) in shard_reranked.into_iter().enumerate() {
            for (local_idx, score) in shard_res {
                let score_int = (score * 10_000_000.0) as i32;
                final_heap.push(RerankCandidate { score_int, shard_idx, local_idx });
                if final_heap.len() > k {
                    final_heap.pop();
                }
            }
        }

        let mut final_results: Vec<RerankCandidate> = final_heap.into_iter().collect();
        // Sort descending (highest score first)
        final_results.sort_unstable_by(|a, b| b.score_int.cmp(&a.score_int));

        final_results.into_iter()
            .map(|c| {
                let score = c.score_int as f32 / 10_000_000.0;
                (self.shards[c.shard_idx].keys[c.local_idx].clone(), score)
            })
            .collect()
    }
}

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

// Generate a deterministic 1536-dimensional unit vector from a text chunk
fn generate_deterministic_embedding(text: &str, dim: usize) -> Vec<f32> {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash_result = hasher.finalize();

    let mut key = [0u8; 32];
    key.copy_from_slice(&hash_result);
    let iv = [0u8; 12];
    let mut cipher = ChaCha20::new(&key.into(), &iv.into());

    let mut buffer = vec![0u8; dim * 4];
    cipher.apply_keystream(&mut buffer);

    let mut floats = vec![0.0f32; dim];
    let mut norm_sq = 0.0f32;
    for i in 0..dim {
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

fn main() {
    println!("📖 Loading Shakespeare corpus...");
    let file = File::open("../shakespeare.txt").expect("Failed to open shakespeare.txt");
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        if let Ok(l) = line {
            let trimmed = l.trim();
            if trimmed.len() > 30 {
                lines.push(trimmed.to_string());
            }
        }
    }
    println!("   Found {} unique real sentences.", lines.len());

    let dim = 1536; 
    let target_vectors = 1_000_000;
    
    // Auto-detect number of shards based on CPU threads
    let num_shards = num_cpus::get();
    println!("⚙️ Initializing Sharded Vector Engine ({} shards) with dim={} for {} vectors...", 
             num_shards, dim, target_vectors);
    let mut engine = ShardedVectorEngine::new(dim, num_shards);

    let start_load = Instant::now();
    println!("🚀 Generating 1,000,000 deterministic float + binary vectors...");
    
    let chunk_size = 50_000;
    for chunk_idx in 0..(target_vectors / chunk_size) {
        let chunk_start = Instant::now();
        let chunk_data: Vec<(AirKey, Vec<u64>, Vec<f32>, usize)> = (0..chunk_size)
            .into_par_iter()
            .map(|i| {
                let global_idx = chunk_idx * chunk_size + i;
                let base_line = &lines[global_idx % lines.len()];
                let unique_text = format!("{}_{}", base_line, global_idx);
                let float_vector = generate_deterministic_embedding(&unique_text, dim);
                
                let mut bit_vector = vec![0u64; dim / 64];
                for idx in 0..dim {
                    if float_vector[idx] >= 0.0 {
                        let block_idx = idx / 64;
                        let bit_idx = idx % 64;
                        bit_vector[block_idx] |= 1 << bit_idx;
                    }
                }
                
                (AirKey(unique_text), bit_vector, float_vector, global_idx)
            })
            .collect();
            
        for (key, bit_vector, float_vector, global_idx) in chunk_data {
            engine.add(key, bit_vector, float_vector, global_idx);
        }
        
        println!("   Generated chunk {}/{} ({} vectors) in {:.2}s", 
                 chunk_idx + 1, target_vectors / chunk_size, chunk_size, chunk_start.elapsed().as_secs_f32());
    }

    println!("💾 Total load/quantization time: {:.2} seconds", start_load.elapsed().as_secs_f32());
    
    // Sizing
    let mut total_bin_bytes = 0;
    let mut total_float_bytes = 0;
    for shard in &engine.shards {
        total_bin_bytes += shard.data.len() * 8;
        total_float_bytes += shard.float_data.len() * 4;
    }
    println!("📊 RAM FOOTPRINT:");
    println!("   Quantized Binary Index: {:.2} MB", total_bin_bytes as f64 / 1024.0 / 1024.0);
    println!("   Raw Float Vectors:      {:.2} MB", total_float_bytes as f64 / 1024.0 / 1024.0);

    // Query
    let query_text = "To be, or not to be, that is the question";
    println!("\n🔍 Running sharded benchmarks for query: '{}'...", query_text);
    let query_float = generate_deterministic_embedding(query_text, dim);
    let query_bin = engine.quantize(&query_float);

    // Warmup
    for _ in 0..5 {
        engine.search_sharded_parallel(&query_bin, 10);
        engine.search_sharded_reranked_parallel(&query_bin, &query_float, 10);
    }

    // Benchmark 1: Shared-Nothing Sharded Parallel Hamming Scan
    let start_bench = Instant::now();
    let num_runs = 100;
    for _ in 0..num_runs {
        engine.search_sharded_parallel(&query_bin, 10);
    }
    let lat_sharded = start_bench.elapsed() / num_runs;
    println!("⏱️  Shared-Nothing Sharded Hamming Scan (1M vectors):");
    println!("   Avg Latency: {:.2} ms", lat_sharded.as_secs_f32() * 1000.0);
    println!("   Throughput:  {:.1} QPS", 1.0 / lat_sharded.as_secs_f32());

    // Benchmark 2: Shared-Nothing Sharded Parallel Hamming + Float Reranking
    let start_bench = Instant::now();
    let mut last_results = Vec::new();
    for _ in 0..num_runs {
        last_results = engine.search_sharded_reranked_parallel(&query_bin, &query_float, 10);
    }
    let lat_sharded_rerank = start_bench.elapsed() / num_runs;
    println!("⏱️  Shared-Nothing Sharded Hamming + Float Reranking (1M vectors):");
    println!("   Avg Latency: {:.2} ms", lat_sharded_rerank.as_secs_f32() * 1000.0);
    println!("   Throughput:  {:.1} QPS", 1.0 / lat_sharded_rerank.as_secs_f32());

    println!("\n🎯 TOP-10 Nearest Neighbors (Float Reranked):");
    for (i, (key, score)) in last_results.iter().enumerate() {
        println!("   {}. Score: {:.4} | Key: {}", i + 1, score, key.0);
    }
}
