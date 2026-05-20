use std::fs::File;
use std::io::Read;
use anyhow::Result;
use serde::Deserialize;
use model2vec_rust::Model2Vec;

#[derive(Debug, Deserialize)]
struct GroundTruthEntry {
    sentence: String,
    embedding: Vec<f32>,
}

fn main() -> Result<()> {
    println!("=== Model2Vec Rust Inference Engine ===");
    
    // Load the model
    println!("Loading Model2Vec model from 'model/' directory...");
    let start_load = std::time::Instant::now();
    let model = Model2Vec::load("model")?;
    println!(
        "Model loaded successfully in {:.2?} (vocab_size = {}, dim = {})",
        start_load.elapsed(),
        model.vocab_size(),
        model.dim()
    );

    // Read ground truth json
    println!("Loading ground_truth.json...");
    let mut file = File::open("ground_truth.json")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let ground_truth: Vec<GroundTruthEntry> = serde_json::from_str(&contents)?;
    println!("Loaded {} entries of ground truth data.", ground_truth.len());

    // We will test both with and without special tokens to see which one matches the Python model2vec
    for add_special_tokens in [true, false] {
        println!("\nTesting correctness with add_special_tokens = {}...", add_special_tokens);
        let mut max_diff = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        let mut total_elements = 0;
        let mut all_match = true;

        for entry in &ground_truth {
            let rust_embedding = model.encode(&entry.sentence, add_special_tokens)?;
            
            if rust_embedding.len() != entry.embedding.len() {
                println!(
                    "Dimension mismatch for sentence '{}': Rust got {}, Python got {}",
                    entry.sentence,
                    rust_embedding.len(),
                    entry.embedding.len()
                );
                all_match = false;
                continue;
            }

            for i in 0..rust_embedding.len() {
                let diff = (rust_embedding[i] - entry.embedding[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                sum_sq_diff += (diff as f64) * (diff as f64);
                total_elements += 1;
            }
        }

        let mse = sum_sq_diff / (total_elements as f64);
        println!("  - Max absolute difference: {:.8}", max_diff);
        println!("  - Mean Squared Error (MSE): {:.8e}", mse);
        
        if all_match && max_diff < 1e-4 {
            println!("  => SUCCESS: Rust implementation matches Python ground truth!");
        } else {
            println!("  => MISMATCH: Difference too high or dimension mismatch occurred.");
        }
    }

    // Demo run of encoding batch
    println!("\nRunning a quick demo batch encoding...");
    let sentences = vec![
        "This is highly optimized Rust code.",
        "Model2Vec on CPU is incredibly lightweight.",
        "We are achieving maximum performance."
    ];

    let start_batch = std::time::Instant::now();
    // Using the flat layout which is most memory-efficient
    let _flat_embeddings = model.encode_batch_flat(&sentences, true)?;
    let elapsed = start_batch.elapsed();
    
    println!(
        "Encoded batch of {} sentences in {:.2?} ({:.2} us per sentence)",
        sentences.len(),
        elapsed,
        (elapsed.as_micros() as f64) / (sentences.len() as f64)
    );

    Ok(())
}
