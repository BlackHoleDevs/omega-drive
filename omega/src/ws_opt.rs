#![allow(dead_code)]

/// Fallback demasking algorithm using standard byte-by-byte XOR loop.
/// This works on all platforms and handles remainder chunks.
pub fn demask_fallback(data: &mut [u8], mask: [u8; 4]) {
    for (i, byte) in data.iter_mut().enumerate() {
        *byte ^= mask[i % 4];
    }
}

/// AVX2-accelerated WebSocket frame demasking.
/// It processes 32 bytes (256 bits) in a single CPU cycle.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn demask_avx2_impl(data: &mut [u8], mask: [u8; 4]) {
    let len = data.len();
    let mask_u32 = u32::from_ne_bytes(mask);
    let mut i = 0;

    use std::arch::x86_64::*;
    
    // Broadcast the 4-byte mask to a 256-bit YMM register.
    // This replicates the 32-bit pattern [m0, m1, m2, m3] 8 times across the 256-bit register.
    let v_mask = _mm256_set1_epi32(mask_u32 as i32);

    // Loop in chunks of 32 bytes
    while i + 32 <= len {
        let ptr = data.as_mut_ptr().add(i);
        // Load 32 bytes from memory (unaligned)
        let v_data = _mm256_loadu_si256(ptr as *const __m256i);
        // Perform bitwise XOR
        let v_res = _mm256_xor_si256(v_data, v_mask);
        // Store 32 bytes back to memory (unaligned)
        _mm256_storeu_si256(ptr as *mut __m256i, v_res);
        i += 32;
    }

    // Process the remaining bytes using the fallback method
    if i < len {
        demask_fallback(&mut data[i..], mask);
    }
}

/// Public entry point for demasking. Automatically detects CPU features at runtime
/// and selects the fastest implementation available.
pub fn demask(data: &mut [u8], mask: [u8; 4]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                demask_avx2_impl(data, mask);
            }
            return;
        }
    }
    
    // Fallback if AVX2 is not supported by CPU or not x86/x86_64
    demask_fallback(data, mask);
}
