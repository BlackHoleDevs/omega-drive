# 🚀 Omega Drive 3.0 — Hybrid Neural Caching & Database Accelerator

[![Language: Rust](https://img.shields.io/badge/Language-Rust-orange.svg?logo=rust&style=flat-square)](https://www.rust-lang.org)
[![Edition: Non-Commercial](https://img.shields.io/badge/Edition-Non--Commercial-blue.svg?style=flat-square)](#%EF%B8%8F-non-commercial-license)
[![Build: Passing](https://img.shields.io/badge/Build-Passing-green.svg?style=flat-square)](#)
[![Security: Neural DNA](https://img.shields.io/badge/Security-Neural%20DNA-red.svg?logo=keycdn&logoColor=white&style=flat-square)](#%EF%B8%8F-neural-cryptography--dna-tampering)

Welcome to the official release of **Omega Drive 3.0 (Non-Commercial Evaluation Edition)**. 

Omega Drive is a state-of-the-art, high-performance hybrid database accelerator and intelligent cache engine. Engineered from the ground up in highly optimized Rust 🦀, Omega Drive combines raw multi-threaded networking with a built-in **Multi-Channel Neural Network (MCNN)** routing cortex to achieve unmatched data-access latency and cryptographic safety.

---

## ⚡ Key Features

* **Hybrid Neural Architecture:** Integrates a neural network routing cortex directly into the memory path, allowing zero-overhead expert selection and intelligent predictive indexing.
* **Extreme Throughput:** Engineered for absolute performance with support for AVX2 instructions, native Linux epoll via Tokio, and optional CUDA-accelerated batch operations, comfortably surpassing **43.5K+ requests per second** with sub-millisecond latencies.
* **Native Redis Protocol compatibility:** Fully compliant with standard RESP protocol semantics, acting as a drop-in replacement or real-time caching layer for existing Redis/Memcached infrastructures.
* **Integrated WebSocket pub/sub Streaming:** Built-in high-speed WebSocket broadcast server running on port `8082` for real-time reactive updates directly to web applications (such as our WooCommerce next-gen accelerators).


---

## ⚖️ Non-Commercial License & Reference Benchmarks

This release (`v1.0.0`) is the **Non-Commercial Evaluation Edition** of Omega Drive. 

* **Active Workers:** Cryptographically capped via the embedded Sentinel Neural DNA to a maximum of **2 parallel worker threads** (active CPU cores).
* **Redistribution:** Granted strictly for non-commercial evaluation, local testing, and educational purposes.
* **Weights File:** Bounded to the cryptographically signed `logic_gate.omm` database weights file.

### 💻 Reference Hardware Specifications
All high-throughput caching and reverse-proxy benchmarks (achieving **43.5K+ requests/sec** for reverse-proxy cache and **13.4M+ operations/sec** for raw Unix Domain Sockets) were conducted and verified locally under a non-commercial environment on the following reference machine:
* **System Model:** Micro-Star International Co., Ltd. WS66 10TKT Notebook
* **OS:** Linux Ubuntu 22.04 LTS (x86_64)
* **Processor:** Intel® Core™ i7-10875H CPU @ 2.30GHz × 16 threads
* **Memory:** 32GB Dual-Channel High-Speed DDR4 RAM

---

## ⚙️ Quick Start

To launch and run the Omega Drive accelerator on a Linux terminal:

### 1. Prepare the License Weights
The database engine requires a signed neural database weights file named `logic_gate.omm` in its working directory to pass the Neural Integrity Check. 

### 2. Launch the Accelerator
Run the pre-compiled `omega` server binary:

```bash
# Start in the foreground
./omega

# Start binding to a custom port and address
./omega --port 6380 --bind 127.0.0.1
```

### 3. Verify the Sentinel Handshake
Upon launching, the engine will perform the Neural Sentinel integrity handshake:

```text
🚀 OMEGA DRIVE 3.0 - HYBRID NEURAL GATEWAY
🧬 Neural License Verified | Active: 2/16 Workers [LITE TIER - 2 Cores Bound]
🌐 Worker 0 online
🌐 Worker 1 online
🚀 [ACCELERATOR] WebSocket Pub/Sub Server active on ws://127.0.0.1:8082 [License Bounds: 2 Cores]
```

---

## 📡 Command Line Options

Customize the daemon's behavior by passing the following flags:

| Flag | Shorthand | Description | Default |
| :--- | :--- | :--- | :--- |
| `--port` | `-p` | TCP Port to bind the database server to | `6380` |
| `--bind` | | Interface IP address to bind | `127.0.0.1` |
| `--workers` | `-w` | Override number of parallel worker threads | *Auto-detected* |
| `--ws-port` | | Port for the WebSocket accelerator streaming | `8082` |
| `--daemonize` | | Run the process in the background as a daemon (`yes` / `no`) | `no` |
| `--device` | `-d` | Target compute hardware (`cpu`, `gpu`, or `hybrid`) | `cpu` |
| `--unixsocket`| | Bind to a local Unix Domain Socket (for zero-latency IPC) | *None* |

> [!WARNING]
> **Experimental GPU Acceleration (`--device gpu` / `hybrid`):** The GPU compute feature is strictly **experimental** in this release. Please note that it **cannot be used on standard Virtual Private Servers (VPS)**. VPS instances generally lack physical GPU passthrough and CUDA driver support, which will cause initialization errors. Use the default `cpu` mode for all standard cloud and VPS deployments.

---

## 💻 Connecting a Client

Because Omega Drive is fully compatible with the standard Redis protocol, you can use any standard client library to perform hyper-fast operations:

### Shell (redis-cli)
```bash
redis-cli -p 6380 PING
# Output: PONG
```

### Python (Single Keys)
```python
import redis

# Connect to Omega Drive Accelerator
client = redis.Redis(host='127.0.0.1', port=6380)

# Set a neural cache value (uses cryptographic cascade path)
client.set('greeting', 'Hello from Omega Drive!')

# Get the decrypted value
print(client.get('greeting').decode('utf-8'))
```

### Python (Bulk MSET / MGET Acceleration)
For massive datasets and high-volume pipelines, Omega Drive features a specialized **Raw Neural Expert path** for bulk operations. By bypassing the Cascade Cryptography layer, `MSET` and `MGET` achieve absolute hardware saturation and maximum raw throughput:

```python
import redis

client = redis.Redis(host='127.0.0.1', port=6380)

# 1. Bulk write via MSET (Fast Neural Expert path)
bulk_data = {
    f"sensor_node_{i}": f"telemetry_data_chunk_payload_{i * 99}"
    for i in range(100)
}
client.mset(bulk_data)
print("✅ Successfully stored 100 bulk chunks via MSET!")

# 2. Bulk read via MGET
keys = [f"sensor_node_{i}" for i in range(5)]
results = client.mget(keys)

for k, v in zip(keys, results):
    print(f"📡 {k} -> {v.decode('utf-8') if v else 'None'}")
```


*For commercial licensing, clustering support, and unlimited high-performance performance tiers, contact the Omega Drive Creator.*
