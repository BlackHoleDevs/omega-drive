# 🚀 OmegaDrive: Hybrid Neural Caching & Vector Search Accelerator

[![Language: Rust](https://img.shields.io/badge/Language-Rust-orange.svg?logo=rust&style=flat-square)](https://www.rust-lang.org)
[![License: MIT / Apache 2.0](https://img.shields.io/badge/License-MIT%20%2F%20Apache%202.0-blue.svg?style=flat-square)](LICENSE.md)
[![Build: Passing](https://img.shields.io/badge/Build-Passing-green.svg?style=flat-square)](#)
[![Performance: 597K+ RPS](https://img.shields.io/badge/Performance-597K%20RPS-brightgreen.svg?logo=speedtest&logoColor=white&style=flat-square)](#-key-performance-benchmarks)
[![Vector Search: 1M Scale](https://img.shields.io/badge/Vector%20Search-1M%20Scale-blue.svg?logo=vector&style=flat-square)](#-hybrid-vector-search-engine)

OmegaDrive is a next-generation, high-performance hybrid memory database and vector search engine written from scratch in highly optimized **Rust** 🦀. Fully compatible with the standard **Redis (RESP)** protocol, OmegaDrive acts as a drop-in database replacement, zero-latency caching accelerator, and ultra-fast vector retrieval engine.

Through a **Shared-Nothing architecture**, CPU-bound SIMD vectorization, and a kernel-space load balancer, OmegaDrive breaks free from the single-threaded bottlenecks of traditional databases to achieve unmatched raw performance and efficiency.

---

## 🧬 Architectural Overview

OmegaDrive is built around three core design principles to maximize hardware saturation:
1. **Shared-Nothing Worker Model:** Rather than relying on a global thread pool or lock manager (which causes CPU cache bouncing and thread lock contention), each CPU core runs its own isolated, single-threaded runtime.
2. **Kernel-Space Load Balancing (`SO_REUSEPORT`):** Inbound TCP/UDS connections are load-balanced at the kernel level directly to the worker event loops, bypassing acceptor bottlenecks.
3. **Contiguous Memory bitstreams:** Keys, Redis Hashes, and Vector embeddings are stored in flat, contiguous memory layouts (`Vec<u8>`, `Vec<f32>`, `Vec<u64>`) to maximize L1/L2 cache prefetching and enable zero-heap-allocation network streaming.

```mermaid
graph TD
    Client[Client Connections] -->|SO_REUSEPORT Load Balancing| Kernel[Linux Kernel Socket Listener]
    Kernel --> Worker0[Worker Core 0]
    Kernel --> Worker1[Worker Core 1]
    Kernel --> WorkerN[Worker Core N]
    
    subgraph Shared-Nothing Workers
        Worker0 -->|Event Loop| RESP0[RESP Parser]
        Worker1 -->|Event Loop| RESP1[RESP Parser]
        WorkerN -->|Event Loop| RESPn[RESP Parser]
    end
    
    RESP0 --> FlatMemory[Flat Contiguous Memory Store]
    RESP1 --> FlatMemory
    RESPn --> FlatMemory
    
    subgraph Memory Architecture
        FlatMemory --> HashStore[Flat Key-Value Hash Engine]
        FlatMemory --> CryptEngine[AVX2 Neural Cryptography Cascade]
        FlatMemory --> VectorStore[Hybrid 1-bit + Float Vector Engine]
    end
```

---

## ⚡ Key Features

* **High-Throughput memtier Benchmarks:** Reaches over **597,000 requests per second** under high concurrency—outperforming Redis and KeyDB by up to **2.92x**.
* **RESP Protocol Compatibility:** Drop-in support for Redis client libraries (`redis-py`, `node-redis`, `redis-cli`, etc.).
* **Dynamic Neural Cascade Cipher:** double-layer cryptographic protection (ChaCha20 + Neural XOR) processed at AVX2 instruction levels, providing wire-speed security.
* **Hybrid Vector Search:** Optimized vector database combining **1-bit Sign Quantization (SRP)** Hamming filter with **AVX2-unrolled Float Reranking**, maintaining **100% recall** at 1,000,000 vectors with **15x speedups**.
* **High-Speed WebSocket Pub/Sub:** Real-time push gateway on port `8082` featuring AVX2 SIMD demasking (up to **15.8x faster** demasking).

---

## 🏎️ Key Performance Benchmarks

All benchmarks were executed on an AMD Ryzen/Intel reference machine running Ubuntu 24.04. See [benchmarks.md](benchmarks.md) for full reproduction commands and configurations.

### 1. High Concurrency Showdown (`memtier_benchmark`)
*Profile: 8 Threads, 32 Connections/Thread (256 clients), 1:1 SET/GET ratio.*

* **Redis v7.2:** `204,261 ops/sec` | Avg. Latency: `1.25 ms`
* **KeyDB v6.3:** `196,037 ops/sec` | Avg. Latency: `1.30 ms`
* **OmegaDrive:** **`597,474 ops/sec` 🚀** | Avg. Latency: **`0.45 ms` ⚡**

### 2. YCSB Workload B (95% Reads, 5% Updates)
*Profile: 12 Threads, 50,000 Transactions.*

* **KeyDB:** `94,876 ops/sec` | READ Avg. Latency: `114.4 μs`
* **Redis:** `104,602 ops/sec` | READ Avg. Latency: `105.8 μs`
* **OmegaDrive:** **`156,739 ops/sec` 🔥** | READ Avg. Latency: **`55.9 μs` ⚡**

### 3. Vector Scaling Speedup (1536 Dimensions)
*Profile: Average latency for 100 queries.*

* **100,000 Vectors:** Standard Float scan `34.47 ms` vs Omega Reranked **`2.00 ms`** (**17.24x Speedup**, **100% Recall**)
* **1,000,000 Vectors:** Standard Float scan `328.40 ms` vs Omega Reranked **`21.66 ms`** (**15.16x Speedup**, **100% Recall**)

---

## ⚙️ Building & Running

### 1. Compile from Source
To enable AVX2, FMA, and CPU-level hardware vectorization, compile with `target-cpu=native`:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### 2. Start the Server
OmegaDrive requires the neural database weights file `logic_gate.omm` in its active working directory to successfully pass the Neural Integrity Handshake.

```bash
# Start in the foreground on default port 6380
./target/release/omega

# Start binding to all interfaces and custom Unix Domain Socket
./target/release/omega --bind 0.0.0.0 --port 6380 --unixsocket /tmp/omega.sock --daemonize yes
```

Upon launching, the engine outputs the verified handshake:
```text
🚀 OMEGA DRIVE 3.0 - HYBRID NEURAL GATEWAY
🧬 Neural License Verified | Active: 16/16 Workers [UNLIMITED PERFORMANCE TIER]
🌐 Worker 0 online
...
🌐 Worker 15 online
🚀 [ACCELERATOR] WebSocket Pub/Sub Server active on ws://127.0.0.1:8082
```

---

## 📡 Command Line Arguments

| Argument | Shorthand | Description | Default |
| :--- | :--- | :--- | :--- |
| `--port` | `-p` | TCP Port to bind the database server to | `6380` |
| `--bind` | | Interface IP address to bind | `127.0.0.1` |
| `--workers` | `-w` | Override number of parallel worker threads | *Auto-detected* |
| `--ws-port` | | Port for the WebSocket accelerator streaming | `8082` |
| `--daemonize` | | Run the process in the background as a daemon (`yes` / `no`) | `no` |
| `--device` | `-d` | Target compute hardware (`cpu`, `gpu`, or `hybrid`) | `cpu` |
| `--unixsocket`| | Bind to a local Unix Domain Socket (for zero-latency IPC) | *None* |

---

## 💻 Code Examples

### 1. Connecting via Standard Redis Client (Python)
Because OmegaDrive implements standard RESP protocol parsing, it acts as a drop-in replacement:

```python
import redis

# Connect to OmegaDrive over loopback port
client = redis.Redis(host='127.0.0.1', port=6380)

# Set a key (stored securely in flat memory layout)
client.set('db_mode', 'unlimited_open_source')

# Retrieve the key
print(client.get('db_mode').decode('utf-8'))
# Output: unlimited_open_source
```

### 2. Native Vector Search Commands (VADD & VSEARCH)
OmegaDrive features native vector database commands (`VADD` and `VSEARCH`). Vectors are automatically quantized to 1-bit representation internally, while the raw floats are stored for exact dot-product reranking.

#### VADD Syntax
```text
VADD <key> <float_1> <float_2> ... <float_dim>
```
*Adds the vector to the database. Returns `+OK`.*

#### VSEARCH Syntax
```text
VSEARCH <top_k> <float_1> <float_2> ... <float_dim>
```
*Searches the database using Coarse Filtering + Exact Float Reranking. Returns a flat RESP Array of `[key_1, score_1 * 10000, key_2, score_2 * 10000, ...]`.*

#### Client Example (Python)
```python
import redis
import random

client = redis.Redis(host='127.0.0.1', port=6380)

# 1. Insert two high-dimensional vectors (e.g. dimension 1536)
vec_a = [random.uniform(-1.0, 1.0) for _ in range(1536)]
vec_b = [random.uniform(-1.0, 1.0) for _ in range(1536)]

client.execute_command("VADD", "embedding_node_a", *vec_a)
client.execute_command("VADD", "embedding_node_b", *vec_b)
print("✅ Vector nodes inserted!")

# 2. Perform a nearest-neighbor VSEARCH (k=2)
query_vec = [random.uniform(-1.0, 1.0) for _ in range(1536)]
results = client.execute_command("VSEARCH", 2, *query_vec)

# Process results
# Flat array returned: [b'key_1', score_1, b'key_2', score_2]
for i in range(0, len(results), 2):
    key = results[i].decode('utf-8')
    score = results[i+1] / 10000.0
    print(f"🎯 Match {i//2 + 1}: {key} (Cosine Similarity Score: {score:.4f})")
```

### 3. Subscribing to the WebSocket Broadcast Gateway
For real-time data streaming and WooCommerce/Next.js reactivity:

```javascript
const WebSocket = require('ws');

// Connect to the WebSocket Pub/Sub port
const ws = new WebSocket('ws://127.0.0.1:8082');

ws.on('open', () => {
  console.log('Connected to OmegaDrive real-time gateway!');
  // Subscribe to channel
  ws.send(JSON.stringify({ action: 'SUBSCRIBE', channel: 'telemetry_stream' }));
});

ws.on('message', (data) => {
  console.log('Received telemetry update:', JSON.parse(data));
});
```

---

## ⚖️ Open Source License

OmegaDrive is open-source software dual-licensed under the **MIT License** and the **Apache License, Version 2.0**.
