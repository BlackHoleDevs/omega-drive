use sysinfo::System;
use dashmap::DashMap;
use std::sync::Arc;
use crate::{AirKey, NeuralValue};

pub fn spawn_telemetry_thread(db: Arc<DashMap<AirKey, NeuralValue, fxhash::FxBuildHasher>>) {
    std::thread::spawn(move || {
        let mut sys = System::new_all();
        if let Ok(pid) = sysinfo::get_current_pid() {
            let mut input = String::new();
            loop {
                input.clear();
                // Waiting for user input in the background
                if std::io::stdin().read_line(&mut input).is_ok() {
                    let trimmed = input.trim();
                    if trimmed == "s" || trimmed == "S" {
                        sys.refresh_all();
                        if let Some(process) = sys.process(pid) {
                            // process.memory() returns bytes in sysinfo 0.30+
                            let memory_mb = process.memory() as f64 / 1024.0 / 1024.0;
                            let cpu_usage = process.cpu_usage();
                            let keys_count = db.len();
                            // Estimating database RAM footprint
                            // DashMap overhead (~32B) + AirKey (~24B) + NeuralValue (~72B) = ~128 bytes per entry
                            let db_size_mb = (keys_count as f64 * 128.0) / 1024.0 / 1024.0;
                            
                            println!("\n📊 --- OMEGADRIVE TELEMETRY REPORT ---");
                            println!("💻 CPU Usage: {:.2}%", cpu_usage);
                            println!("🧠 RAM Usage: {:.2} MB", memory_mb);
                            println!("📦 Database:  {} keys active (Est. {:.2} MB in RAM)", keys_count, db_size_mb);
                            println!("--------------------------------------\n");
                        } else {
                            println!("⚠️ [TELEMETRY] Failed to read process metrics. Process might have terminated.");
                        }
                    }
                }
            }
        } else {
            println!("⚠️ [TELEMETRY] Failed to acquire current PID. Interactive telemetry disabled.");
        }
    });
}
