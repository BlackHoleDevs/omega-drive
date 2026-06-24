extern "C" __global__ void forward_router_batch(
    const unsigned char* __restrict__ inputs_u8, // [batch_size, 64] RAW BYTES
    const float* __restrict__ w1,          // [32, 64]
    const float* __restrict__ b1,          // [32]
    const float* __restrict__ w2,          // [100, 32]
    const float* __restrict__ b2,          // [100]
    const unsigned char* __restrict__ keys, // [100, 64]
    const float* __restrict__ tier_config, // [1] Neural Throttle
    unsigned char* __restrict__ outputs,   // [batch_size, 64]
    int* __restrict__ best_indices,        // [batch_size]
    int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // ECO-FRIENDLY NEURAL THROTTLING (God Mode = 0 ns)
    unsigned int throttle_ns = (unsigned int)tier_config[0];
    if (throttle_ns > 0) {
        #if __CUDA_ARCH__ >= 700
            __nanosleep(throttle_ns);
        #else
            long long start = clock64();
            while(clock64() - start < (long long)throttle_ns) {}
        #endif
    }

    // 1. In-place normalization (u8 -> f32) inside registers
    float x[64];
    const unsigned char* my_input = &inputs_u8[tid * 64];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        x[i] = (float)my_input[i] * 0.0039215686f; // 1/255
    }

    // 2. Layer 1: 64 -> 32 (ReLU)
    float h[32];
    for (int i = 0; i < 32; i++) {
        float sum = b1[i];
        #pragma unroll
        for (int j = 0; j < 64; j++) {
            sum += w1[i * 64 + j] * x[j];
        }
        h[i] = (sum > 0.0f) ? sum : 0.0f;
    }

    // 3. Layer 2: 32 -> 100
    int best_idx = 0;
    float max_val = -1.0e30f;
    for (int i = 0; i < 100; i++) {
        float sum = b2[i];
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            sum += w2[i * 32 + j] * h[j];
        }
        if (sum > max_val) {
            max_val = sum;
            best_idx = i;
        }
    }
    best_indices[tid] = best_idx;

    // 4. XOR Engine
    const unsigned char* key = &keys[best_idx * 64];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        outputs[tid * 64 + i] = my_input[i] ^ key[i];
    }
}

extern "C" __global__ void reconstruct_batch(
    const unsigned char* __restrict__ compressed,
    const int* __restrict__ spec_indices,
    const unsigned char* __restrict__ keys,
    unsigned char* __restrict__ outputs,
    int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    int spec_idx = spec_indices[tid];
    const unsigned char* key = &keys[spec_idx * 64];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        outputs[tid * 64 + i] = compressed[tid * 64 + i] ^ key[i];
    }
}
