#![allow(clippy::type_complexity, clippy::needless_range_loop)]
use safetensors::tensor::SafeTensors;
use std::fs::File;
use std::io::Read;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
const PTX: &str = include_str!("airdb_kernel.ptx");

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Device { Cpu, Gpu, Hybrid }

pub struct McnnModel {
    router_w1: Vec<f32>, router_b1: Vec<f32>, router_w2: Vec<f32>, pub router_b2: Vec<f32>,
    pub bit_keys: Vec<[u8; 64]>, #[allow(dead_code)] pub worker_limit: usize, pub has_avx2: bool,
    #[cfg(feature = "cuda")]
    pub cuda_device: Option<Arc<CudaDevice>>,
    #[cfg(feature = "cuda")]
    gpu_weights: Option<Arc<GpuWeights>>,
}

#[cfg(feature = "cuda")]
struct GpuWeights {
    w1: cudarc::driver::CudaSlice<f32>, b1: cudarc::driver::CudaSlice<f32>,
    w2: cudarc::driver::CudaSlice<f32>, b2: cudarc::driver::CudaSlice<f32>,
    keys: cudarc::driver::CudaSlice<u8>, tier_config: cudarc::driver::CudaSlice<f32>,
}

impl McnnModel {
    pub fn load(path: &str, _device_type: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let data_part = if buffer.len() > 32 {
            &buffer[..buffer.len() - 32]
        } else {
            &buffer
        };

        let tensors = SafeTensors::deserialize(data_part)?;
        let bit_keys = Self::load_bit_keys(&tensors);
        let worker_limit = 9999;

        let to_vec = |name: &str| -> Vec<f32> {
            let view = tensors.tensor(name).unwrap();
            unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const f32, view.data().len() / 4).to_vec() }
        };

        let router_w1 = to_vec("swarm.router.0.weight");
        let router_b1 = to_vec("swarm.router.0.bias");
        let router_w2 = to_vec("swarm.router.2.weight");
        let router_b2 = to_vec("swarm.router.2.bias");

        #[cfg(feature = "cuda")]
        let mut cuda_device = None;
        #[cfg(feature = "cuda")]
        let mut gpu_weights = None;
        
        #[cfg(feature = "cuda")]
        if let Device::Gpu | Device::Hybrid = _device_type {
            if let Ok(dev) = CudaDevice::new(0) {
                if dev.load_ptx(Ptx::from_src(PTX), "airdb", &["forward_router_batch", "reconstruct_batch"]).is_ok() {
                    let w1_gpu = dev.htod_copy(router_w1.clone())?;
                    let b1_gpu = dev.htod_copy(router_b1.clone())?;
                    let w2_gpu = dev.htod_copy(router_w2.clone())?;
                    let b2_gpu = dev.htod_copy(router_b2.clone())?;
                    let mut flat_keys = Vec::with_capacity(100 * 64);
                    for k in &bit_keys { flat_keys.extend_from_slice(k); }
                    let keys_gpu = dev.htod_copy(flat_keys)?;
                    let tier_config_gpu = dev.htod_copy(vec![0.0f32])?;
                    gpu_weights = Some(Arc::new(GpuWeights { w1: w1_gpu, b1: b1_gpu, w2: w2_gpu, b2: b2_gpu, keys: keys_gpu, tier_config: tier_config_gpu }));
                    cuda_device = Some(dev);
                }
            }
        }

        let has_avx2 = {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            { is_x86_feature_detected!("avx2") }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            { false }
        };

        Ok(Self { 
            router_w1, router_b1, router_w2, router_b2, bit_keys, worker_limit, has_avx2, 
            #[cfg(feature = "cuda")]
            cuda_device, 
            #[cfg(feature = "cuda")]
            gpu_weights 
        })
    }

    fn load_bit_keys(tensors: &SafeTensors) -> Vec<[u8; 64]> {
        let view = tensors.tensor("swarm.bit_keys").unwrap();
        let data = unsafe { std::slice::from_raw_parts(view.data().as_ptr() as *const f32, view.data().len() / 4) };
        (0..100).map(|i| {
            let mut key = [0u8; 64]; let expert_floats = &data[i * 512 .. (i + 1) * 512];
            for j in 0..64 {
                let mut byte = 0u8; for b in 0..8 { if expert_floats[j * 8 + b] > 0.5 { byte |= 1 << b; } }
                key[j] = byte;
            }
            key
        }).collect()
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn forward_avx2(&self, input: &[u8; 64]) -> (usize, [u8; 64]) {
        let mut x = [0.0f32; 64]; let inv_255 = _mm256_set1_ps(1.0 / 255.0);
        for i in (0..64).step_by(8) {
            let m128_i = _mm_set_epi64x(0, *(input.as_ptr().add(i) as *const u64) as i64);
            _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(m128_i)), inv_255));
        }
        let mut h = [0.0f32; 32];
        for i in 0..32 {
            let mut acc = _mm256_setzero_ps(); let row_ptr = self.router_w1.as_ptr().add(i * 64);
            for j in (0..64).step_by(8) { acc = _mm256_fmadd_ps(_mm256_loadu_ps(row_ptr.add(j)), _mm256_loadu_ps(x.as_ptr().add(j)), acc); }
            h[i] = (self.horizontal_sum(acc) + self.router_b1[i]).max(0.0);
        }
        let mut best_idx = 0; let mut max_val = f32::MIN;
        for i in 0..100 {
            let mut acc = _mm256_setzero_ps(); let row_ptr = self.router_w2.as_ptr().add(i * 32);
            for j in (0..32).step_by(8) { acc = _mm256_fmadd_ps(_mm256_loadu_ps(row_ptr.add(j)), _mm256_loadu_ps(h.as_ptr().add(j)), acc); }
            let res = self.horizontal_sum(acc) + self.router_b2[i];
            if res > max_val { max_val = res; best_idx = i; }
        }
        (best_idx, self.forward_with_expert_inline(best_idx, input))
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline(always)]
    unsafe fn horizontal_sum(&self, v: __m256) -> f32 {
        let v_high = _mm256_extractf128_ps(v, 1); let v_low = _mm256_extractf128_ps(v, 0);
        let v_sum = _mm_add_ps(v_high, v_low); let v_shuf = _mm_movehl_ps(v_sum, v_sum);
        let v_sum2 = _mm_add_ps(v_sum, v_shuf); let v_shuf2 = _mm_shuffle_ps(v_sum2, v_sum2, 1);
        _mm_cvtss_f32(_mm_add_ss(v_sum2, v_shuf2))
    }

    pub fn forward_single(&self, chunk: &[u8; 64]) -> (usize, [u8; 64]) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        { if self.has_avx2 { unsafe { return self.forward_avx2(chunk); } } }
        let (idx, _) = self.forward_fallback(chunk); (idx, self.forward_with_expert_scalar(idx, chunk))
    }

    pub fn forward_raw(&self, input: &[u8]) -> (usize, Vec<[u8; 64]>) {
        if input.is_empty() { let (idx, out) = self.forward_single(&[0u8; 64]); return (idx, vec![out]); }
        let chunks_count = input.len().div_ceil(64);
        #[cfg(feature = "cuda")]
        if chunks_count >= 1024 && self.cuda_device.is_some() { if let Ok(res) = self.forward_gpu_async(input) { return res; } }
        let mut results = Vec::with_capacity(chunks_count);
        
        // SEQUENTIAL NEURAL STREAMS (Zero Overhead)
        let (best_idx, _) = self.forward_single(&input[..input.len().min(64)].try_into().unwrap_or([0u8; 64]));
        
        for i in 0..chunks_count {
            let mut row = [0u8; 64];
            let start = i * 64;
            let end = (start + 64).min(input.len());
            row[..end - start].copy_from_slice(&input[start..end]);
            results.push(self.forward_with_expert_scalar(best_idx, &row));
        }

        (best_idx, results)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_gpu_async(&self, input: &[u8]) -> Result<(usize, Vec<[u8; 64]>), Box<dyn std::error::Error>> {
        let dev = self.cuda_device.as_ref().unwrap(); let weights = self.gpu_weights.as_ref().unwrap();
        let chunks_count = input.len().div_ceil(64);
        let mut padded_input = vec![0u8; chunks_count * 64]; padded_input[..input.len()].copy_from_slice(input);
        let inp_gpu = dev.htod_copy(padded_input)?; let mut out_gpu = dev.alloc_zeros::<u8>(chunks_count * 64)?;
        let mut idx_gpu = dev.alloc_zeros::<i32>(chunks_count)?;
        let fwd = dev.get_func("airdb", "forward_router_batch").unwrap();
        let cfg = LaunchConfig::for_num_elems(chunks_count as u32);
        let stream = dev.fork_default_stream()?;
        unsafe { fwd.launch_on_stream(&stream, cfg, (&inp_gpu, &weights.w1, &weights.b1, &weights.w2, &weights.b2, &weights.keys, &weights.tier_config, &mut out_gpu, &mut idx_gpu, chunks_count as i32))?; }
        dev.synchronize()?;
        let results_raw = dev.dtoh_sync_copy(&out_gpu)?; let indices = dev.dtoh_sync_copy(&idx_gpu)?;
        let mut results = Vec::with_capacity(chunks_count);
        for i in 0..chunks_count { let mut chunk = [0u8; 64]; chunk.copy_from_slice(&results_raw[i*64..(i+1)*64]); results.push(chunk); }
        Ok((indices[0] as usize, results))
    }

    fn forward_with_expert_scalar(&self, best_idx: usize, row: &[u8; 64]) -> [u8; 64] {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        { if self.has_avx2 { unsafe { return self.forward_with_expert_inline(best_idx, row); } } }
        let key = &self.bit_keys[best_idx]; let mut output = [0u8; 64];
        for i in 0..64 { output[i] = row[i] ^ key[i]; }
        output
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn forward_with_expert_inline(&self, best_idx: usize, row: &[u8; 64]) -> [u8; 64] {
        let key = &self.bit_keys[best_idx]; let mut output = [0u8; 64];
        let k_v = _mm256_loadu_si256(key.as_ptr() as *const __m256i); let k_v2 = _mm256_loadu_si256(key.as_ptr().add(32) as *const __m256i);
        let s_v = _mm256_loadu_si256(row.as_ptr() as *const __m256i); let s_v2 = _mm256_loadu_si256(row.as_ptr().add(32) as *const __m256i);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, _mm256_xor_si256(s_v, k_v));
        _mm256_storeu_si256(output.as_mut_ptr().add(32) as *mut __m256i, _mm256_xor_si256(s_v2, k_v2));
        output
    }

    fn forward_fallback(&self, input: &[u8; 64]) -> (usize, [u8; 64]) {
        let mut x = [0.0f32; 64]; for i in 0..64 { x[i] = input[i] as f32 / 255.0; }
        let mut h = [0.0f32; 32];
        for i in 0..32 {
            let mut sum = self.router_b1[i]; let row = &self.router_w1[i*64..(i+1)*64];
            for j in 0..64 { sum += row[j] * x[j]; }
            h[i] = sum.max(0.0);
        }
        let mut best_idx = 0; let mut max_val = f32::MIN;
        for i in 0..100 {
            let mut sum = self.router_b2[i]; let row = &self.router_w2[i*32..(i+1)*32];
            for j in 0..32 { sum += row[j] * h[j]; }
            if sum > max_val { max_val = sum; best_idx = i; }
        }
        (best_idx, [0u8; 64])
    }

    pub fn reconstruct(&self, best_idx: usize, compressed: &[u8; 64]) -> [u8; 64] { self.forward_with_expert_scalar(best_idx, compressed) }
    pub fn reconstruct_raw(&self, best_idx: usize, chunks: &[crate::ChunkData], original_len: usize) -> Vec<u8> {
        #[cfg(feature = "cuda")]
        if chunks.len() >= 1024 && self.cuda_device.is_some() { if let Ok(res) = self.reconstruct_gpu_ext(best_idx, chunks, original_len) { return res; } }
        
        // DIRECT SEQUENTIAL RECONSTRUCTION (Zero Overhead)
        let mut out = Vec::with_capacity(chunks.len() * 64);
        for chunk in chunks {
            out.extend_from_slice(&self.reconstruct(best_idx, &chunk.0));
        }
        
        out.truncate(original_len); out
    }
    #[cfg(feature = "cuda")]
    pub fn reconstruct_gpu_ext(&self, best_idx: usize, chunks: &[crate::ChunkData], original_len: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let dev = self.cuda_device.as_ref().unwrap(); let weights = self.gpu_weights.as_ref().unwrap();
        let chunks_count = chunks.len();
        let mut flat_input = Vec::with_capacity(chunks_count * 64);
        for c in chunks { flat_input.extend_from_slice(&c.0); }
        let inp_gpu = dev.htod_copy(flat_input)?; let mut out_gpu = dev.alloc_zeros::<u8>(chunks_count * 64)?;
        let idx_gpu = dev.htod_copy(vec![best_idx as i32; chunks_count])?;
        let fwd = dev.get_func("airdb", "reconstruct_batch").unwrap();
        let cfg = LaunchConfig::for_num_elems(chunks_count as u32);
        unsafe { fwd.launch(cfg, (&inp_gpu, &idx_gpu, &weights.keys, &mut out_gpu, chunks_count as i32))?; }
        let mut results = dev.dtoh_sync_copy(&out_gpu)?; results.truncate(original_len); Ok(results)
    }

    pub fn forward_cascade(&self, input: &[u8]) -> (usize, Vec<[u8; 64]>) {
        if input.is_empty() {
            let (idx, out) = self.forward_single(&[0u8; 64]);
            return (idx, vec![out]);
        }
        
        let chunks_count = input.len().div_ceil(64);
        let mut results = Vec::with_capacity(chunks_count);
        
        // 1. Swarm routing to get the best_idx expert key
        let (best_idx, _) = self.forward_single(&input[..input.len().min(64)].try_into().unwrap_or([0u8; 64]));
        let expert_key = &self.bit_keys[best_idx];
        
        // 2. Setup ChaCha20 cipher using first 32 bytes of expert_key as key and next 12 bytes as nonce
        use chacha20::ChaCha20;
        use chacha20::cipher::{KeyIvInit, StreamCipher};
        let key = chacha20::Key::from_slice(&expert_key[0..32]);
        let nonce = chacha20::Nonce::from_slice(&expert_key[32..44]);
        let mut cipher = ChaCha20::new(key, nonce);
        
        // 3. Encrypt the entire input buffer with ChaCha20
        let mut temp_buf = input.to_vec();
        cipher.apply_keystream(&mut temp_buf);
        
        // 4. Run Neural XOR expert encryption over the ChaCha20 encrypted chunks
        for i in 0..chunks_count {
            let mut row = [0u8; 64];
            let start = i * 64;
            let end = (start + 64).min(temp_buf.len());
            row[..end - start].copy_from_slice(&temp_buf[start..end]);
            results.push(self.forward_with_expert_scalar(best_idx, &row));
        }
        
        (best_idx, results)
    }

    pub fn reconstruct_cascade(&self, best_idx: usize, chunks: &[crate::ChunkData], original_len: usize) -> Vec<u8> {
        let expert_key = &self.bit_keys[best_idx];
        
        // 1. Reverse the Neural XOR expert encryption
        let mut decrypted_chunks = Vec::with_capacity(chunks.len() * 64);
        for chunk in chunks {
            decrypted_chunks.extend_from_slice(&self.reconstruct(best_idx, &chunk.0));
        }
        decrypted_chunks.truncate(original_len);
        
        // 2. Setup ChaCha20 cipher and decrypt
        use chacha20::ChaCha20;
        use chacha20::cipher::{KeyIvInit, StreamCipher};
        let key = chacha20::Key::from_slice(&expert_key[0..32]);
        let nonce = chacha20::Nonce::from_slice(&expert_key[32..44]);
        let mut cipher = ChaCha20::new(key, nonce);
        
        cipher.apply_keystream(&mut decrypted_chunks);
        decrypted_chunks
    }
}

