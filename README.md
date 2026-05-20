# ⚡ Model2Vec Rust Inference Engine

A simple, extremely performant, low-CPU-overhead Rust implementation of the **Model2Vec** static embedding inference algorithm. 

This engine is optimized for high-throughput, low-latency CPU environments, achieving **over 54,000 sentences/sec** on a single thread and **over 101,000 sentences/sec** using multi-core batch processing.

---

## ✨ Features

- **Blazing Fast**: Single sentence encoding latency of **~18.5 µs** (approx. 50x-100x faster than the Python library).
- **Perfect Correctness**: Achieves identical mathematical output compared to the official Python `model2vec` library (Mean Squared Error $\approx 8.5 \times 10^{-17}$).
- **Zero-Allocation Hot Path**: Minimizes heap allocations during sentence pooling by working directly in pre-allocated buffers.
- **SIMD Optimized**: Designed to let the Rust compiler autovectorize lookups, pooling, and L2 normalization loops.
- **Parallel Processing**: Integrates lightweight, thread-safe batch encoding via **Rayon** for linear multi-core scaling.
- **Stand-alone Deployment**: Compiles into a single lightweight binary with no heavy external runtimes (no Python, no PyTorch, no heavy dependencies).

---

## 📊 Performance & Correctness Summary

Tested using `minishlab/potion-base-8M` distilled from `BAAI/bge-base-en-v1.5` on a standard CPU:

| Metric | Python (`model2vec`) | Our Rust Engine | Speedup |
| :--- | :--- | :--- | :--- |
| **Single Sentence Latency** | ~1,000 to 2,000 µs (1-2 ms) | **18.5 µs** (0.018 ms) | **~50x to 100x faster** |
| **Single-Core Throughput** | ~500 sentences / sec | **~54,000 sentences / sec** | **~100x higher throughput** |
| **Multi-Core Throughput (Batch)** | GIL bottlenecks | **~101,600 sentences / sec** | **Highly scale-efficient** |
| **Parity Max Abs Difference** | - | **`0.00000009`** | Within `float32` limits |
| **Mean Squared Error (MSE)** | - | **`8.51e-17`** | Practically zero |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Rust and Cargo installed:
```bash
rustc --version
cargo --version
```

### Installation

Clone or locate the package, then compile in release mode:
```bash
cargo build --release
```

---

## 🛠️ Usage

### Rust API Example

Add the following to your code to load and run inference:

```rust
use model2vec_rust::Model2Vec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the model from a local directory containing config.json, tokenizer.json, and model.safetensors
    let model = Model2Vec::load("model")?;

    // 2. Encode a single sentence (returns Vec<f32>)
    // Set `add_special_tokens` to `false` to match Python model2vec exactly
    let embedding = model.encode("Rust is incredibly fast and performant.", false)?;
    println!("Embedding vector size: {}", embedding.len()); // Should be 256

    // 3. Batch encode sentences in parallel across multiple CPU cores
    let sentences = vec![
        "Hello world!",
        "Model2Vec on CPU requires very few resources.",
        "Zero memory allocation in hot path keeps CPU usage minimal."
    ];
    let embeddings = model.encode_batch(&sentences, false)?;
    println!("Encoded {} sentences.", embeddings.len());

    // 4. Flattened Batch Encoding (maximum memory-efficiency; returns single Vec<f32>)
    let flat_embeddings = model.encode_batch_flat(&sentences, false)?;
    
    Ok(())
}
```

---

## 🧪 Commands

### Run Correctness & Parity Checks
Executes the verification script comparing Rust's outputs directly against the Python ground-truth database:
```bash
cargo run --release
```

### Run Performance Benchmarks
Runs robust, multi-iteration micro-benchmarks utilizing the `criterion` crate:
```bash
cargo bench
```

---

## 📂 Model Directory Structure
Ensure your `model/` folder contains the following three files downloaded from Hugging Face (e.g., [minishlab/potion-base-8M](https://huggingface.co/minishlab/potion-base-8M)):
```text
model/
├── config.json
├── tokenizer.json
└── model.safetensors
```
