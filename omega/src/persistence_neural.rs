use std::fs::OpenOptions;
use std::io::Write;
use std::sync::mpsc::Receiver;
use std::time::{Instant, Duration};

#[derive(Clone, Debug)]
pub enum PersistenceCommand {
    Set(String, Vec<u8>),
    Del(String),
    HmSet(String, Vec<(String, Vec<u8>)>),
    VAdd(String, Vec<f32>),
    FlushDb,
}

impl PersistenceCommand {
    pub fn to_resp(&self) -> Vec<u8> {
        match self {
            PersistenceCommand::Set(key, val) => {
                let mut buf = Vec::with_capacity(32 + key.len() + val.len());
                buf.extend_from_slice(b"*3\r\n$3\r\nSET\r\n");
                buf.extend_from_slice(format!("${}\r\n", key.len()).as_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(b"\r\n");
                buf.extend_from_slice(format!("${}\r\n", val.len()).as_bytes());
                buf.extend_from_slice(val);
                buf.extend_from_slice(b"\r\n");
                buf
            }
            PersistenceCommand::Del(key) => {
                let mut buf = Vec::with_capacity(24 + key.len());
                buf.extend_from_slice(b"*2\r\n$3\r\nDEL\r\n");
                buf.extend_from_slice(format!("${}\r\n", key.len()).as_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(b"\r\n");
                buf
            }
            PersistenceCommand::HmSet(key, fields) => {
                let num_args = 2 + fields.len() * 2;
                let mut buf = Vec::with_capacity(64 + key.len() + fields.len() * 32);
                buf.extend_from_slice(format!("*{}\r\n$5\r\nHMSET\r\n", num_args).as_bytes());
                buf.extend_from_slice(format!("${}\r\n", key.len()).as_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(b"\r\n");
                for (f_key, f_val) in fields {
                    buf.extend_from_slice(format!("${}\r\n", f_key.len()).as_bytes());
                    buf.extend_from_slice(f_key.as_bytes());
                    buf.extend_from_slice(b"\r\n");
                    buf.extend_from_slice(format!("${}\r\n", f_val.len()).as_bytes());
                    buf.extend_from_slice(f_val);
                    buf.extend_from_slice(b"\r\n");
                }
                buf
            }
            PersistenceCommand::VAdd(key, floats) => {
                let num_args = 2 + floats.len();
                let mut buf = Vec::with_capacity(64 + key.len() + floats.len() * 10);
                buf.extend_from_slice(format!("*{}\r\n$4\r\nVADD\r\n", num_args).as_bytes());
                buf.extend_from_slice(format!("${}\r\n", key.len()).as_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(b"\r\n");
                for f in floats {
                    let f_str = f.to_string();
                    buf.extend_from_slice(format!("${}\r\n", f_str.len()).as_bytes());
                    buf.extend_from_slice(f_str.as_bytes());
                    buf.extend_from_slice(b"\r\n");
                }
                buf
            }
            PersistenceCommand::FlushDb => {
                b"*1\r\n$7\r\nFLUSHDB\r\n".to_vec()
            }
        }
    }
}

pub struct NeuralPersistenceAgent {
    w1: [[f32; 20]; 8],
    b1: [f32; 8],
    w2: [f32; 8],
    b2: f32,
    learning_rate: f32,
    seed: u32,
}

impl NeuralPersistenceAgent {
    pub fn new() -> Self {
        let mut seed = 1337u32;
        let mut w1 = [[0.0f32; 20]; 8];
        let mut b1 = [0.0f32; 8];
        let mut w2 = [0.0f32; 8];
        
        // Initialize weights using small random values
        for i in 0..8 {
            b1[i] = Self::random_f32(&mut seed) * 0.1;
            w2[i] = Self::random_f32(&mut seed) * 0.1;
            for j in 0..20 {
                w1[i][j] = Self::random_f32(&mut seed) * 0.1;
            }
        }
        let b2 = Self::random_f32(&mut seed) * 0.1;

        Self {
            w1,
            b1,
            w2,
            b2,
            learning_rate: 0.005,
            seed,
        }
    }

    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        if x == 0 { x = 1; }
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    fn random_f32(seed: &mut u32) -> f32 {
        let u = Self::xorshift32(seed);
        (u as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    pub fn forward(&self, state: &[f32; 20]) -> (f32, [f32; 8]) {
        let mut hidden = [0.0f32; 8];
        for i in 0..8 {
            let mut z = self.b1[i];
            for j in 0..20 {
                z += state[j] * self.w1[i][j];
            }
            hidden[i] = 1.0 / (1.0 + (-z).exp()); // Sigmoid
        }

        let mut z2 = self.b2;
        for i in 0..8 {
            z2 += hidden[i] * self.w2[i];
        }
        let prob = 1.0 / (1.0 + (-z2).exp()); // Sigmoid output probability

        (prob, hidden)
    }

    pub fn select_action(&mut self, prob: f32) -> bool {
        let r: f32 = (Self::xorshift32(&mut self.seed) as f32) / (u32::MAX as f32);
        // Stochastic action sampling based on output probability
        r < prob
    }

    pub fn update_weights(
        &mut self,
        state: &[f32; 20],
        hidden: &[f32; 8],
        action: f32,
        prob: f32,
        reward: f32,
    ) {
        // Gradient of log likelihood * reward
        let loss_grad = (action - prob) * reward;

        // Update output weights
        self.b2 += loss_grad * self.learning_rate;
        for i in 0..8 {
            self.w2[i] += loss_grad * hidden[i] * self.learning_rate;
        }

        // Backprop to hidden layers
        for i in 0..8 {
            let hidden_grad = loss_grad * self.w2[i] * hidden[i] * (1.0 - hidden[i]);
            self.b1[i] += hidden_grad * self.learning_rate;
            for j in 0..20 {
                self.w1[i][j] += hidden_grad * state[j] * self.learning_rate;
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn pin_thread_to_core(core_id: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(core_id, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    println!("📍 [NPC] Thread pinned to CPU Core {}", core_id);
}

#[cfg(not(target_os = "linux"))]
fn pin_thread_to_core(core_id: usize) {
    println!("📍 [NPC] Core pinning not supported on this OS. Running standard thread.");
}

pub fn run_persistence_worker(rx: Receiver<PersistenceCommand>, dump_path: String) {
    pin_thread_to_core(0);

    let mut file = loop {
        match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&dump_path)
        {
            Ok(f) => break f,
            Err(e) => {
                println!("❌ [NPC] Failed to open {}: {:?}. Retrying in 1s...", dump_path, e);
                std::thread::sleep(Duration::from_secs(1));
            }
        }
    };

    println!("💾 [NPC] Neural Persistence Controller started using file: {}", dump_path);

    let mut agent = NeuralPersistenceAgent::new();
    
    // Feature tracking states
    let mut history = [0.0f32; 15];
    let mut last_write_time_ms = 0.0f32;
    let mut last_fsync_duration_ms = 0.0f32;
    let mut last_time = Instant::now();
    let mut batch_buffer: Vec<PersistenceCommand> = Vec::new();

    loop {
        // Collect batch from channel (non-blocking unless buffer is empty)
        while let Ok(cmd) = rx.try_recv() {
            batch_buffer.push(cmd);
        }

        let queue_len = batch_buffer.len();

        if queue_len == 0 {
            // No commands to write; wait a bit and loop
            std::thread::sleep(Duration::from_micros(200));
            continue;
        }

        // Shift history window
        for i in (1..15).rev() {
            history[i] = history[i - 1];
        }
        history[0] = queue_len as f32;

        let elapsed = last_time.elapsed().as_secs_f32() * 1000.0;
        last_time = Instant::now();

        // Construct 20 input features:
        // x_0: current queue length
        // x_1: elapsed time since last loop tick
        // x_2: write time of the last batch (ms)
        // x_3: sync duration of the last batch (ms)
        // x_4: history average
        // x_5..x_19: rolling history of queue lengths
        let mut state = [0.0f32; 20];
        state[0] = queue_len as f32;
        state[1] = elapsed;
        state[2] = last_write_time_ms;
        state[3] = last_fsync_duration_ms;
        
        let mut history_sum = 0.0f32;
        for i in 0..15 {
            state[5 + i] = history[i];
            history_sum += history[i];
        }
        state[4] = history_sum / 15.0;

        // Neural inference decision
        let (prob, hidden) = agent.forward(&state);
        let commit = agent.select_action(prob);

        if commit {
            let write_start = Instant::now();
            let mut write_payload = Vec::new();
            for cmd in &batch_buffer {
                write_payload.extend_from_slice(&cmd.to_resp());
            }

            // Write and fsync with built-in retry-on-failure loop
            let mut success = false;
            let mut retry_count = 0;
            while !success {
                match file.write_all(&write_payload) {
                    Ok(_) => {
                        match file.sync_all() {
                            Ok(_) => {
                                success = true;
                            }
                            Err(e) => {
                                retry_count += 1;
                                println!("⚠️ [NPC] fsync failed (retry {}): {:?}. Retrying in 50ms...", retry_count, e);
                                std::thread::sleep(Duration::from_millis(50));
                            }
                        }
                    }
                    Err(e) => {
                        retry_count += 1;
                        println!("⚠️ [NPC] Write failed (retry {}): {:?}. Retrying in 50ms...", retry_count, e);
                        std::thread::sleep(Duration::from_millis(50));
                        // Re-open file handler just in case it got broken
                        if let Ok(f) = OpenOptions::new().create(true).append(true).open(&dump_path) {
                            file = f;
                        }
                    }
                }
            }

            let write_duration = write_start.elapsed().as_secs_f32() * 1000.0;
            last_write_time_ms = write_duration;
            last_fsync_duration_ms = write_duration * 0.8; // Estimated portion

            // Compute reward (target: minimize wait time while keeping writes efficient)
            let backlog_penalty = - (queue_len as f32) * 0.1;
            let write_penalty = - (write_duration) * 0.05;
            let reward = backlog_penalty + write_penalty;

            // Policy gradient weight update
            agent.update_weights(&state, &hidden, 1.0, prob, reward);

            // Log details occasionally
            if queue_len > 100 {
                println!(
                    "💾 [NPC] Committed {} commands in {:.2}ms (Reward: {:.2}, Prob: {:.2})",
                    queue_len, write_duration, reward, prob
                );
            }

            // Clear the committed batch
            batch_buffer.clear();
        } else {
            // Action = 0 (Wait). Update policy weights.
            // Penalty for waiting is proportional to queue length (encourages committing when backlog increases)
            let reward = - (queue_len as f32) * 0.05;
            agent.update_weights(&state, &hidden, 0.0, prob, reward);

            // Wait brief moment before next evaluation
            std::thread::sleep(Duration::from_micros(500));
        }
    }
}
