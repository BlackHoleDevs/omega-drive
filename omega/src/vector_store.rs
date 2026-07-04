#![allow(dead_code)]
use crate::AirKey;
use std::sync::RwLock;

pub struct VectorEngine {
    keys: Vec<AirKey>,
    // Flat memory layout: continuous slice of u64s containing all packed binary vectors
    data: Vec<u64>,
    // Flat memory layout for raw float vectors (used during exact float reranking)
    float_data: Vec<f32>,
    dim: usize,
    num_blocks: usize,
}

impl VectorEngine {
    pub fn new(dim: usize) -> Self {
        assert!(dim % 64 == 0, "Dimension must be a multiple of 64 for binary vector quantization");
        let num_blocks = dim / 64;
        Self {
            keys: Vec::new(),
            data: Vec::new(),
            float_data: Vec::new(),
            dim,
            num_blocks,
        }
    }

    /// Quantize a slice of f32 floats into a packed array of u64 bits.
    /// Standard 1-bit quantization (sign random projection): bit is 1 if val >= 0.0, else 0.
    pub fn quantize(&self, floats: &[f32]) -> Vec<u64> {
        let mut packed = vec![0u8; self.num_blocks * 8];
        let limit = floats.len().min(self.dim);
        
        for i in 0..limit {
            if floats[i] >= 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                packed[byte_idx] |= 1 << bit_idx;
            }
        }

        // Convert u8 slice to u64 values
        let mut u64_data = vec![0u64; self.num_blocks];
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
    pub fn add(&mut self, key: AirKey, bit_vector: Vec<u64>, float_vector: Vec<f32>) {
        assert_eq!(bit_vector.len(), self.num_blocks, "Invalid bit vector dimension");
        // Pad or truncate float vector to dim
        let mut floats = vec![0.0f32; self.dim];
        let len = float_vector.len().min(self.dim);
        floats[..len].copy_from_slice(&float_vector[..len]);

        self.keys.push(key);
        self.data.extend_from_slice(&bit_vector);
        self.float_data.extend_from_slice(&floats);
    }

    /// Search for TOP-K nearest vectors using Hamming Distance (XOR + POPCNT).
    /// Returns a list of (AirKey, Distance) sorted by distance ascending (lower distance = more similar).
    pub fn search(&self, query: &[u64], k: usize) -> Vec<(AirKey, u32)> {
        if query.len() != self.num_blocks || self.keys.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(self.keys.len());

        // Liniowy scan ciągłej pamięci - idealne dla L1/L2 cache CPU
        for i in 0..self.keys.len() {
            let start = i * self.num_blocks;
            let end = start + self.num_blocks;
            let candidate = &self.data[start..end];

            let mut distance = 0u32;
            for j in 0..self.num_blocks {
                distance += (candidate[j] ^ query[j]).count_ones();
            }

            results.push((self.keys[i].clone(), distance));
        }

        // Sort by distance ascending
        results.sort_unstable_by_key(|&(_, dist)| dist);
        results.truncate(k);
        results
    }

    /// Search for TOP-K nearest vectors using Coarse Filter (Hamming) + Exact Rerank (Float Dot Product).
    /// Returns a list of (AirKey, exact float score) sorted by score descending (higher is more similar).
    pub fn search_reranked(&self, query_bin: &[u64], query_float: &[f32], k: usize) -> Vec<(AirKey, f32)> {
        if query_bin.len() != self.num_blocks || self.keys.is_empty() {
            return Vec::new();
        }

        let num_vectors = self.keys.len();
        // Dynamically scale coarse-filter candidate pool size (1% of N, minimum 1000)
        let r = (num_vectors / 100).max(1000).min(num_vectors);

        let mut scores: Vec<(usize, u32)> = (0..num_vectors)
            .map(|idx| {
                let start = idx * self.num_blocks;
                let end = start + self.num_blocks;
                let candidate = &self.data[start..end];
                let mut dist = 0u32;
                for j in 0..self.num_blocks {
                    dist += (query_bin[j] ^ candidate[j]).count_ones();
                }
                (idx, dist)
            })
            .collect();

        // Perform linear quick-select to keep only the top R candidate indices
        if r > 0 {
            scores.select_nth_unstable_by_key(r - 1, |&(_, dist)| dist);
            scores.truncate(r);
        }

        // Rerank those R candidates using exact dot product
        let mut reranked: Vec<(usize, f32)> = scores.iter()
            .map(|&(idx, _)| {
                let start = idx * self.dim;
                let end = start + self.dim;
                let candidate = &self.float_data[start..end];
                let score = dot_product(query_float, candidate);
                (idx, score)
            })
            .collect();

        // Sort descending by exact dot product score
        reranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        reranked.iter()
            .take(k)
            .map(|&(idx, score)| (self.keys[idx].clone(), score))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.keys.len()
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

pub struct SharedVectorStore {
    inner: RwLock<VectorEngine>,
}

impl SharedVectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            inner: RwLock::new(VectorEngine::new(dim)),
        }
    }

    pub fn add(&self, key: AirKey, bit_vector: Vec<u64>, float_vector: Vec<f32>) {
        if let Ok(mut store) = self.inner.write() {
            store.add(key, bit_vector, float_vector);
        }
    }

    pub fn quantize(&self, floats: &[f32]) -> Vec<u64> {
        if let Ok(store) = self.inner.read() {
            store.quantize(floats)
        } else {
            Vec::new()
        }
    }

    pub fn search(&self, query: &[u64], k: usize) -> Vec<(AirKey, u32)> {
        if let Ok(store) = self.inner.read() {
            store.search(query, k)
        } else {
            Vec::new()
        }
    }

    pub fn search_reranked(&self, query_bin: &[u64], query_float: &[f32], k: usize) -> Vec<(AirKey, f32)> {
        if let Ok(store) = self.inner.read() {
            store.search_reranked(query_bin, query_float, k)
        } else {
            Vec::new()
        }
    }

    pub fn len(&self) -> usize {
        if let Ok(store) = self.inner.read() {
            store.len()
        } else {
            0
        }
    }
}
