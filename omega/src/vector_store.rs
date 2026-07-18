#![allow(dead_code)]
use crate::AirKey;
use std::sync::RwLock;
use std::collections::BinaryHeap;
use rayon::prelude::*;

#[derive(Clone)]
struct BlockSignature {
    s_high: Vec<u64>,
    s_low: Vec<u64>,
    num_superpositions: u32,
    start_idx: usize,
    end_idx: usize,
}

struct CoreShard {
    keys: Vec<AirKey>,
    data: Vec<u64>,
    float_data: Vec<f32>,
    blocks: Vec<BlockSignature>,
}

impl CoreShard {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            data: Vec::new(),
            float_data: Vec::new(),
            blocks: Vec::new(),
        }
    }

    fn rebuild_blocks(&mut self, num_blocks: usize) {
        let num_vectors = self.keys.len();
        let block_size = 128;
        let mut expected_blocks = num_vectors / block_size;
        if num_vectors % block_size != 0 {
            expected_blocks += 1;
        }

        // If the number of blocks matches, only check if the last block needs an update
        if self.blocks.len() == expected_blocks {
            if let Some(last_block) = self.blocks.last_mut() {
                let actual_end = num_vectors;
                if last_block.end_idx != actual_end {
                    let sig = Self::compute_signature(&self.data, last_block.start_idx, actual_end, num_blocks);
                    *last_block = sig;
                }
                return;
            }
        }

        // Otherwise, rebuild starting from the current blocks length
        let start_block = self.blocks.len();
        for b_idx in start_block..expected_blocks {
            let start_idx = b_idx * block_size;
            let end_idx = ((b_idx + 1) * block_size).min(num_vectors);
            let sig = Self::compute_signature(&self.data, start_idx, end_idx, num_blocks);
            self.blocks.push(sig);
        }
    }

    fn compute_signature(data: &[u64], start_idx: usize, end_idx: usize, num_blocks: usize) -> BlockSignature {
        let w = end_idx - start_idx;
        let mut s_high = vec![0u64; num_blocks];
        let mut s_low = vec![0u64; num_blocks];
        let mut num_superpositions = 0u32;

        let high_threshold = (w * 7) / 10;
        let low_threshold = (w * 3) / 10;

        for word_idx in 0..num_blocks {
            let mut bit_counts = [0usize; 64];
            for v_idx in start_idx..end_idx {
                let word = data[v_idx * num_blocks + word_idx];
                for b in 0..64 {
                    if (word & (1u64 << b)) != 0 {
                        bit_counts[b] += 1;
                    }
                }
            }

            let mut sh = 0u64;
            let mut sl = 0u64;
            for b in 0..64 {
                if bit_counts[b] >= high_threshold {
                    sh |= 1u64 << b;
                } else if bit_counts[b] <= low_threshold {
                    sl |= 1u64 << b;
                }
            }
            s_high[word_idx] = sh;
            s_low[word_idx] = sl;
            num_superpositions += (!(sh | sl)).count_ones();
        }

        BlockSignature {
            s_high,
            s_low,
            num_superpositions,
            start_idx,
            end_idx,
        }
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
        other.score_int.cmp(&self.score_int)
    }
}

impl PartialOrd for RerankCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct SharedVectorStore {
    shards: Vec<RwLock<CoreShard>>,
    dim: usize,
    num_blocks: usize,
    write_counter: std::sync::atomic::AtomicUsize,
}

impl SharedVectorStore {
    pub fn new(dim: usize) -> Self {
        assert!(dim % 64 == 0, "Dimension must be a multiple of 64 for binary vector quantization");
        let num_shards = num_cpus::get().max(4);
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(RwLock::new(CoreShard::new()));
        }
        Self {
            shards,
            dim,
            num_blocks: dim / 64,
            write_counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Quantize a slice of f32 floats into a packed array of u64 bits.
    /// Standard 1-bit quantization (sign random projection): bit is 1 if val >= 0.0, else 0.
    pub fn quantize(&self, floats: &[f32]) -> Vec<u64> {
        let mut u64_data = vec![0u64; self.num_blocks];
        let limit = floats.len().min(self.dim);
        
        let mut packed = vec![0u8; self.num_blocks * 8];
        for i in 0..limit {
            if floats[i] >= 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                packed[byte_idx] |= 1 << bit_idx;
            }
        }

        for i in 0..self.num_blocks {
            let start = i * 8;
            let mut val = 0u64;
            for j in 0..8 {
                val |= (packed[start + j] as u64) << (j * 8);
            }
            u64_data[i] = val;
        }
        u64_data
    }

    /// Add a binary vector and original float vector to the engine.
    pub fn add(&self, key: AirKey, bit_vector: Vec<u64>, float_vector: Vec<f32>) {
        let idx = self.write_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let shard_idx = idx % self.shards.len();
        if let Ok(mut shard) = self.shards[shard_idx].write() {
            // Pad or truncate float vector to dim
            let mut floats = vec![0.0f32; self.dim];
            let len = float_vector.len().min(self.dim);
            floats[..len].copy_from_slice(&float_vector[..len]);

            shard.keys.push(key);
            shard.data.extend_from_slice(&bit_vector);
            shard.float_data.extend_from_slice(&floats);

            // Build/update blocks dynamically
            shard.rebuild_blocks(self.num_blocks);
        }
    }

    /// Search for TOP-K nearest vectors using Hamming Distance with HSSP pruning.
    pub fn search(&self, query: &[u64], k: usize) -> Vec<(AirKey, u32)> {
        // 1. Scan each shard in parallel using Rayon (HSSP pruned)
        let shard_results: Vec<Vec<(u32, usize)>> = self.shards
            .par_iter()
            .map(|shard_lock| {
                if let Ok(shard) = shard_lock.read() {
                    let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(k + 1);
                    
                    let query_low: Vec<u64> = query.iter().map(|&w| !w).collect();
                    
                    let mut block_evals: Vec<(u32, usize)> = shard.blocks.iter().enumerate().map(|(b_idx, block)| {
                        let mut min_dist = 0u32;
                        for j in 0..self.num_blocks {
                            min_dist += (query[j] ^ block.s_high[j]).count_ones();
                            min_dist += (query_low[j] ^ block.s_low[j]).count_ones();
                        }
                        let min_dist = min_dist.saturating_sub(block.num_superpositions);
                        (min_dist, b_idx)
                    }).collect();
                    
                    block_evals.sort_unstable_by_key(|&(d, _)| d);
                    
                    for &(min_dist, b_idx) in &block_evals {
                        if heap.len() >= k {
                            if min_dist >= heap.peek().unwrap().0 {
                                break;
                            }
                        }
                        
                        let block = &shard.blocks[b_idx];
                        for i in block.start_idx..block.end_idx {
                            let start = i * self.num_blocks;
                            let candidate = &shard.data[start..start + self.num_blocks];
                            let mut dist = 0u32;
                            for j in 0..self.num_blocks {
                                dist += (candidate[j] ^ query[j]).count_ones();
                            }
                            if heap.len() < k {
                                heap.push((dist, i));
                            } else if dist < heap.peek().unwrap().0 {
                                heap.push((dist, i));
                                heap.pop();
                            }
                        }
                    }
                    heap.into_iter().collect()
                } else {
                    Vec::new()
                }
            })
            .collect();

        // 2. Merge local top-K results from all shards
        let mut merged = BinaryHeap::with_capacity(k + 1);
        for (shard_idx, shard_res) in shard_results.into_iter().enumerate() {
            for (dist, local_idx) in shard_res {
                merged.push((dist, shard_idx, local_idx));
                if merged.len() > k {
                    merged.pop();
                }
            }
        }

        let mut final_results: Vec<(u32, usize, usize)> = merged.into_iter().collect();
        final_results.sort_unstable_by_key(|&(dist, _, _)| dist);

        if let Ok(shards) = self.shards.iter().map(|s| s.read()).collect::<Result<Vec<_>, _>>() {
            final_results.into_iter()
                .map(|(dist, shard_idx, local_idx)| {
                    (shards[shard_idx].keys[local_idx].clone(), dist)
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Search for TOP-K nearest vectors using Coarse Filter (HSSP Hamming) + Exact Rerank (Float Dot Product).
    pub fn search_reranked(&self, query_bin: &[u64], query_float: &[f32], k: usize) -> Vec<(AirKey, f32)> {
        let total_vectors = self.len();
        if total_vectors == 0 {
            return Vec::new();
        }

        let r = (total_vectors / 500).max(1000).min(total_vectors);
        let r_per_shard = (r / self.shards.len()).max(k);

        // 1. Sequential Hamming distance scan to get local top R candidates per shard with HSSP pruning
        let shard_candidates: Vec<Vec<(u32, usize)>> = self.shards
            .iter()
            .map(|shard_lock| {
                if let Ok(shard) = shard_lock.read() {
                    let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(r_per_shard + 1);
                    
                    let query_low: Vec<u64> = query_bin.iter().map(|&w| !w).collect();
                    
                    let mut block_evals: Vec<(u32, usize)> = shard.blocks.iter().enumerate().map(|(b_idx, block)| {
                        let mut min_dist = 0u32;
                        for j in 0..self.num_blocks {
                            min_dist += (query_bin[j] ^ block.s_high[j]).count_ones();
                            min_dist += (query_low[j] ^ block.s_low[j]).count_ones();
                        }
                        let min_dist = min_dist.saturating_sub(block.num_superpositions);
                        (min_dist, b_idx)
                    }).collect();
                    
                    block_evals.sort_unstable_by_key(|&(d, _)| d);
                    
                    for &(min_dist, b_idx) in &block_evals {
                        if heap.len() >= r_per_shard {
                            if min_dist >= heap.peek().unwrap().0 {
                                break;
                            }
                        }
                        
                        let block = &shard.blocks[b_idx];
                        for i in block.start_idx..block.end_idx {
                            let start = i * self.num_blocks;
                            let candidate = &shard.data[start..start + self.num_blocks];
                            let mut dist = 0u32;
                            for j in 0..self.num_blocks {
                                dist += (candidate[j] ^ query_bin[j]).count_ones();
                            }
                            if heap.len() < r_per_shard {
                                heap.push((dist, i));
                            } else if dist < heap.peek().unwrap().0 {
                                heap.push((dist, i));
                                heap.pop();
                            }
                        }
                    }
                    heap.into_iter().collect()
                } else {
                    Vec::new()
                }
            })
            .collect();

        // 2. Sequential exact float reranking within each shard
        let shard_reranked: Vec<Vec<(usize, f32)>> = self.shards
            .iter()
            .zip(shard_candidates.iter())
            .map(|(shard_lock, candidates)| {
                if let Ok(shard) = shard_lock.read() {
                    let mut heap = BinaryHeap::with_capacity(k + 1);
                    for &(_, local_idx) in candidates {
                        let start = local_idx * self.dim;
                        let candidate = &shard.float_data[start..start + self.dim];
                        let score = dot_product(query_float, candidate);
                        let score_int = (score * 10_000_000.0) as i32;
                        let candidate_item = RerankCandidate { score_int, shard_idx: 0, local_idx };
                        if heap.len() < k {
                            heap.push(candidate_item);
                        } else if score_int > heap.peek().unwrap().score_int {
                            heap.push(candidate_item);
                            heap.pop();
                        }
                    }
                    heap.into_iter()
                        .map(|c| (c.local_idx, c.score_int as f32 / 10_000_000.0))
                        .collect()
                } else {
                    Vec::new()
                }
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
        final_results.sort_unstable_by(|a, b| b.score_int.cmp(&a.score_int));

        if let Ok(shards) = self.shards.iter().map(|s| s.read()).collect::<Result<Vec<_>, _>>() {
            final_results.into_iter()
                .map(|c| {
                    let score = c.score_int as f32 / 10_000_000.0;
                    (shards[c.shard_idx].keys[c.local_idx].clone(), score)
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn len(&self) -> usize {
        let mut total = 0;
        for shard in &self.shards {
            if let Ok(s) = shard.read() {
                total += s.keys.len();
            }
        }
        total
    }

    pub fn clear(&self) {
        for shard_lock in &self.shards {
            if let Ok(mut shard) = shard_lock.write() {
                shard.keys.clear();
                shard.data.clear();
                shard.float_data.clear();
                shard.blocks.clear();
            }
        }
        self.write_counter.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

// Manually unrolled 8x accumulator loop to allow AVX2/FMA auto-vectorization
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
