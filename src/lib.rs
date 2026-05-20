use std::fs::File;
use std::io::Read;
use std::path::Path;
use anyhow::{anyhow, Result};
use serde::Deserialize;
use tokenizers::Tokenizer;
use safetensors::SafeTensors;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub normalize: bool,
    pub hidden_dim: usize,
}

pub struct Model2Vec {
    tokenizer: Tokenizer,
    embeddings: Vec<f32>,
    vocab_size: usize,
    dim: usize,
    normalize: bool,
}

impl Model2Vec {
    pub fn load<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        
        // Load config
        let config_path = model_dir.join("config.json");
        let mut config_file = File::open(&config_path)
            .map_err(|e| anyhow!("Failed to open config.json: {}", e))?;
        let mut config_str = String::new();
        config_file.read_to_string(&mut config_str)?;
        let config: ModelConfig = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer.json: {}", e))?;

        // Load safetensors
        let weights_path = model_dir.join("model.safetensors");
        let weights_bytes = std::fs::read(&weights_path)
            .map_err(|e| anyhow!("Failed to read model.safetensors: {}", e))?;
        
        let safetensor = SafeTensors::deserialize(&weights_bytes)
            .map_err(|e| anyhow!("Failed to deserialize safetensors: {}", e))?;

        let (_, view) = safetensor.tensors()
            .into_iter()
            .find(|(name, _)| name == "embeddings")
            .ok_or_else(|| anyhow!("Failed to find 'embeddings' tensor in safetensors"))?;

        let shape = view.shape();
        if shape.len() != 2 {
            return Err(anyhow!("Embeddings tensor must be 2D, got shape {:?}", shape));
        }
        let vocab_size = shape[0];
        let dim = shape[1];
        
        if dim != config.hidden_dim {
            return Err(anyhow!(
                "Dimension mismatch: safetensors has dim {}, config has hidden_dim {}",
                dim, config.hidden_dim
            ));
        }

        // Copy bytes to aligned Vec<f32>
        let data_bytes = view.data();
        if data_bytes.len() != vocab_size * dim * 4 {
            return Err(anyhow!(
                "Data size mismatch: expected {} bytes, got {}",
                vocab_size * dim * 4,
                data_bytes.len()
            ));
        }

        let mut embeddings = vec![0.0f32; vocab_size * dim];
        for (i, chunk) in data_bytes.chunks_exact(4).enumerate() {
            embeddings[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }

        Ok(Self {
            tokenizer,
            embeddings,
            vocab_size,
            dim,
            normalize: config.normalize,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Single-threaded encoding of a single sentence.
    /// To be extremely performant, it uses a pre-allocated/provided buffer to avoid any allocation.
    pub fn encode_to_buf(&self, text: &str, add_special_tokens: bool, out: &mut [f32]) -> Result<()> {
        if out.len() != self.dim {
            return Err(anyhow!("Output buffer size must match model dimension {}", self.dim));
        }
        
        // Tokenize
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        
        let token_ids = encoding.get_ids();
        
        // Zero out the buffer
        for val in out.iter_mut() {
            *val = 0.0;
        }

        if token_ids.is_empty() {
            return Ok(());
        }

        let mut count = 0;
        for &id in token_ids {
            let idx = id as usize;
            if idx < self.vocab_size {
                let offset = idx * self.dim;
                let emb_slice = &self.embeddings[offset..offset + self.dim];
                for i in 0..self.dim {
                    out[i] += emb_slice[i];
                }
                count += 1;
            }
        }

        if count > 0 {
            if self.normalize {
                let mut sum_sq = 0.0f32;
                for i in 0..self.dim {
                    sum_sq += out[i] * out[i];
                }
                if sum_sq > 0.0 {
                    let norm = sum_sq.sqrt();
                    let inv_norm = 1.0 / norm;
                    for i in 0..self.dim {
                        out[i] *= inv_norm;
                    }
                }
            } else {
                let inv_count = 1.0 / (count as f32);
                for i in 0..self.dim {
                    out[i] *= inv_count;
                }
            }
        }

        Ok(())
    }

    /// High-performance encoding of a single sentence (allocates the return vector).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.dim];
        self.encode_to_buf(text, add_special_tokens, &mut out)?;
        Ok(out)
    }

    /// Parallel batch encoding of sentences using Rayon.
    /// This splits the sentences array across threads and performs zero-allocation encoding per sentence.
    pub fn encode_batch(&self, sentences: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<f32>>> {
        use rayon::prelude::*;
        
        sentences.par_iter()
            .map(|&text| self.encode(text, add_special_tokens))
            .collect()
    }

    /// Flattened batch encoding (returns a single continuous Vec<f32> representing the 2D matrix of embeddings).
    /// This is even faster as it avoids allocating multiple sub-vectors.
    pub fn encode_batch_flat(&self, sentences: &[&str], add_special_tokens: bool) -> Result<Vec<f32>> {
        use rayon::prelude::*;
        
        let n = sentences.len();
        let mut flat_embeddings = vec![0.0f32; n * self.dim];
        
        flat_embeddings.par_chunks_exact_mut(self.dim)
            .zip(sentences.par_iter())
            .try_for_each(|(buf, &text)| {
                self.encode_to_buf(text, add_special_tokens, buf)
            })?;
            
        Ok(flat_embeddings)
    }
}
